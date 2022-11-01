import yaml
import json
import argparse
import os
from datetime import datetime

from PIL import Image
import numpy as np
import labelbox
from labelbox.data.annotation_types import Geometry
from labelbox.data.serialization import COCOConverter
from skimage.measure import regionprops
import cv2

from real_dmg_quadrants import divide_sign_mask_quadrants


dmg_weights = {
    'graffiti': 1.0,
    'cracked': 1.0,
    'obscured_physical': 1.0,
    'missing_sections_holes': 1.0,
    'stickers': 1.0,
    'other_vandalism': 1.0,
    'rust_or_other_aging': 1.0,
    'chipped_paint': 1.0,
    'fading_discoloring_0.5': 0.5,  # Perhaps should have two: 0.6666 and 0.3333?
    'dirt': 1.0,
    'obscured_dark_shadow_0.8': 0.0
}

parser = argparse.ArgumentParser(description="Convert Labelbox annotations from `labels.json` to COCO format.")
parser.add_argument(
    "-d",
    "--debug-viz",
    action="store_true",
    help="Debug visualisation mode."
)
parser.add_argument(
    "--gtsdb-train",
    action="store_true",
    help="Handle GTSDB training set data from `labels_gtsdb_train.json` and `_single_annotations_train.coco.json`."
)
parser.add_argument("--gtsdb-test",
    action="store_true",
    help="Handle GTSDB test set data from `labels_gtsdb_test.json` and `_single_annotations_test.coco.json`."
)
args = parser.parse_args()


def write_image_coco(labels_dict, coco_img_id, lbox_img_id, img_dims):
    # img_dims format: (height, width)
    if next((img for img in labels_dict['images'] if img['id'] == coco_img_id), None) is None:
        labels_dict['images'].append({
            "id": coco_img_id,
            "license": 1,
            "file_name": f"{lbox_img_id}.png",
            "height": img_dims[0],
            "width": img_dims[1],
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

def write_label_coco(labels_dict, label_id, coco_img_id, lbox_img_id, img_dims, bounding_axes,
                     total_dmg, sector_dmgs, damage_type="labelbox_real", copy_axes=False):
    # Bounding axes format (copy_axes=False): (x_left, x_right, y_top, y_bottom)
    # Damage sectors format: [tl, tr, bl, br]
    # img_dims format: (height, width)
    axes = bounding_axes
    if next((img for img in labels_dict['images'] if img['id'] == coco_img_id), None) is None:
        labels_dict['images'].append({
            "id": coco_img_id,
            "license": 1,
            "file_name": f"{lbox_img_id}.png",
            "height": img_dims[0],
            "width": img_dims[1],
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    labels_dict['annotations'].append({
        "id": label_id,
        "image_id": coco_img_id,
        "category_id": 1,
        "bbox": axes if copy_axes else [axes[0], axes[2], axes[1] - axes[0], axes[3] - axes[2]],
        "area": axes[2] * axes[3] if copy_axes else (axes[1] - axes[0]) * (axes[3] - axes[2]),
        "segmentation": [],
        "iscrowd": 0,
        "damage": total_dmg,
        "damage_type": damage_type,
        "sector_damage": sector_dmgs
    })

def get_image_label(label, colors, out_dir):
    image_np = label.data.value
    img_id = label.uid
    Image.fromarray(image_np.astype(np.uint8)).save(f"{out_dir}/{img_id}.png")

    blank_image = Image.new('RGB', (image_np.shape[1], image_np.shape[0]), (0, 0, 0))
    if args.debug_viz:
        # Draw the annotations onto the source image
        for ii, annotation in enumerate(label.annotations):
            if isinstance(annotation.value, Geometry):
                image_np = annotation.value.draw(canvas=image_np,
                                                color=colors[annotation.name],
                                                thickness=5)

                ## DEBUG
                ann = annotation.value.draw(color=colors[annotation.name], thickness=5)
                Image.fromarray(ann.astype(np.uint8)).save(f"{out_dir}/{ii}_{annotation.extra['feature_id']}_{img_id}.png")
                ##

                ## DEBUG
                blank_image = annotation.value.draw(canvas=(None if ii == 0 else blank_image),
                                                    color=colors[annotation.name],
                                                    thickness=5)
                ##
        Image.fromarray(blank_image.astype(np.uint8)).save(f"{out_dir}/annotated_{annotation.extra['feature_id']}_{img_id}.png")
    return image_np

def image_masks_dict(label, colors):
    image_masks = {}
    for annotation in label.annotations:
        is_sign = annotation.name == "Traffic Sign Mask"
        if isinstance(annotation.value, Geometry):
            image_masks[annotation.extra['feature_id']] = {
                'mask': annotation.value.draw(color=colors[annotation.name], thickness=5),
                'is_sign': True,
                'damages': [],
                'combined_bbox': None,
                'total_dmg': None,
                'quad_dmgs': None  # [tl, tr, bl, br]
            } if is_sign else {
                'mask': annotation.value.draw(color=colors[annotation.name], thickness=5),
                'feature_id': annotation.extra['feature_id'],
                'is_sign': False,
                'damage_type': annotation.classifications[0].value.answer.name
            }
    return image_masks

def get_mask_properties(mask):
    regions = regionprops(mask)
    for props in regions:
        cent = props.centroid
        cy, cx = cent[0:2]
        area = props.area  # Area of region pixels, not area of bounding box
        bbox = props.bbox
        # print(bbox)  ##
        miny, minx = bbox[0:2]
        if len(bbox) == 6:
            maxy, maxx = bbox[3:5]
        elif len(bbox) == 4:
            maxy, maxx = bbox[2:4]
        ## DEBUG
        # mask[miny, :] = (0, 255, 0)
        # mask[maxy, :] = (0, 255, 0)
        # mask[:, minx] = (0, 255, 0)
        # mask[:, maxx] = (0, 255, 0)
        ##
        return miny, maxy, minx, maxx, area, cy, cx

def main():
    with open("api_key.yaml", "r") as ymlfile:
        key = yaml.load(ymlfile, Loader=yaml.FullLoader)
    a_token = key['label_token']
    if not args.gtsdb_train and not args.gtsdb_test:
        p_id = key['project_id']
    else:
        p_id = key['project_id_gtsdb_train'] if args.gtsdb_train else key['project_id_gtsdb_test']
    lb = labelbox.Client(api_key=a_token)

    # Get the project
    project = lb.get_project(p_id)
    # Export image and text data as an annotation generator:
    labels = project.label_generator()

    labels = labels.as_list()
    # Create a mapping for the colors
    # hex_to_rgb = lambda hex_color: tuple(
    #     int(hex_color[i + 1:i + 3], 16) for i in (0, 2, 4))
    # colors = {
    #     tool.name: hex_to_rgb(tool.color)
    #     for tool in labelbox.OntologyBuilder.from_project(project).tools
    # }
    colors = {  # NOTE: Colour value must be either 0 or 255
        'Traffic Sign Mask': (0, 0, 255),
        'Damage Mask': (255, 0, 0),
    }

    out_dir = f"dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.mkdir(out_dir)
    labels_path = os.path.join(out_dir, "_single_annotations.coco.json")
    labels_file = open(labels_path, "w")

    # mask_path = f"./{out_dir}/masks/"
    # image_path = f"./{out_dir}/images/"
    # coco_labels = COCOConverter.serialize_instances(
    #     labels,
    #     image_root=image_path,
    #     mask_root=mask_path,
    #     ignore_existing_data=True
    # )

    classes = ['traffic_sign']
    labels_dict = {'categories': [], 'images': [], 'annotations': []}
    labels_dict['categories'] += [{'id': 0, 'name': 'signs', 'supercategory': "none"}]
    labels_dict['categories'] += [{'id': ii + 1, 'name': str(c), 'supercategory': "signs"} for ii, c in enumerate(sorted(classes))]

    all_masks = {}
    if args.debug_viz:
        vis_images = {}

    # Label: https://github.com/Labelbox/labelbox-python/blob/develop/labelbox/data/annotation_types/label.py
    print("Retrieving images and saving label masks...")
    for label in labels:
        # if label.uid != "cl81ana824ww1071xgtgp6tjy":
        #     continue
        print(f"  Retrieving image for: {label.uid}", end='\r')
        value = get_image_label(label, colors, out_dir)
        if args.debug_viz:
            vis_images[label.uid] = value
        print(f"  Retrieving masks for: {label.uid}", end='\r')
        all_masks[label.uid] = image_masks_dict(label, colors)

    print(end="\n\r")
    print("Processing labels...")
    gtsdb_labels = None
    if args.gtsdb_train and args.gtsdb_test:
        raise argparse.ArgumentError("Cannot use both --gtsdb-train and --gtsdb-test simultaneously")
    elif args.gtsdb_train:
        labels_string = "labels_gtsdb_train.json"
        with open("_single_annotations_train.coco.json", "r") as f:
            gtsdb_labels = json.load(f)
    elif args.gtsdb_test:
        labels_string = "labels_gtsdb_test.json"
        with open("_single_annotations_test.coco.json", "r") as f:
            gtsdb_labels = json.load(f)
    else:
        labels_string = "labels.json"
    with open(labels_string, "r") as f:  # Manually downloaded from export on Labelbox website
        labels_json = json.load(f)

    id_mapping = {}
    curr_ann_id = 0
    for image in labels_json:
        if image['ID'] not in id_mapping:
            num = len(id_mapping)
            id_mapping[image['ID']] = num
        coco_img_id = id_mapping[image['ID']]
        lbox_img_id = image['ID']
        
        # Images that were 'skipped' in Labelbox are those either with no signs or only damage values of 0
        if gtsdb_labels is not None and image['Skipped'] is True:
            image_gtsdb = next((img for img in gtsdb_labels['images'] if img['file_name'] == image['External ID']), None)
            if image_gtsdb is None:
                raise ValueError(f"Could not find image {image['External ID']} in GTSDB labels")
            image_gtsdb_shape = (image_gtsdb['height'], image_gtsdb['width'])

            # Copy GTSDB labels with added damage values of 0
            annotations_gtsdb = [ann for ann in gtsdb_labels['annotations'] if ann['image_id'] == image_gtsdb['id']]
            if len(annotations_gtsdb) == 0:
                write_image_coco(labels_dict, coco_img_id, lbox_img_id, image_gtsdb_shape)
            for ann in annotations_gtsdb:
                write_label_coco(
                    labels_dict,
                    curr_ann_id,
                    coco_img_id,
                    lbox_img_id,
                    image_gtsdb_shape,
                    ann['bbox'],
                    0.0,
                    [0.0, 0.0, 0.0, 0.0],
                    copy_axes=True
                )
                curr_ann_id += 1
        else:
            # Link damage masks to sign masks
            relations = image['Label']['relationships']
            for relation in relations:
                target_sign = relation['data']['target']
                source_dmg = relation['data']['source']
                all_masks[image['ID']][target_sign]['damages'].append(all_masks[image['ID']][source_dmg])

            # Make sure image details are always written into labels
            if len(image['Label']['objects']) == 0:  # FIXME: Get real image shape instead of placeholder (-1, -1)
                write_image_coco(labels_dict, coco_img_id, lbox_img_id, (-1, -1))

            # Find bbox for each combined mask
            for mask_id in all_masks[image['ID']]:
                if all_masks[image['ID']][mask_id]['is_sign']:
                    # Get bbox for combined mask and create combined mask itself
                    sign_mask = all_masks[image['ID']][mask_id]['mask']
                    sign_mask_cv = cv2.cvtColor(np.array(sign_mask), cv2.COLOR_RGB2BGR)
                    ##
                    # cv2.imshow('sign_mask_cv', sign_mask_cv)
                    # cv2.waitKey(0)
                    ##
                    try:
                        min_miny, max_maxy, min_minx, max_maxx, sign_mask_area, _, _ = get_mask_properties(sign_mask)
                    except (ValueError, TypeError):
                        raise ValueError(f"Could not find mask for {image['External ID']} with ID {mask_id}")

                    total_sign_area = sign_mask_area
                    weighted_dmg_area = 0
                    for dmg in all_masks[image['ID']][mask_id]['damages']:
                        dmg_mask = cv2.cvtColor(np.array(dmg['mask']), cv2.COLOR_RGB2BGR)
                        sign_mask_cv = cv2.bitwise_or(sign_mask_cv, dmg_mask)
                        ##
                        # cv2.imshow('dmg_mask_cv', dmg_mask)
                        # cv2.imshow('sign_mask_cv', sign_mask_cv)
                        # cv2.waitKey(0)
                        ##

                        # Calculate total damage
                        dmg_mask_gray = cv2.cvtColor(dmg_mask, cv2.COLOR_BGR2GRAY)
                        miny, maxy, minx, maxx, dmg_mask_area, _, _ = get_mask_properties(dmg_mask_gray)
                        min_miny = min(miny, min_miny)
                        max_maxy = max(maxy, max_maxy)
                        min_minx = min(minx, min_minx)
                        max_maxx = max(maxx, max_maxx)
                        total_sign_area += dmg_mask_area
                        weighted_dmg_area += dmg_mask_area * dmg_weights[dmg['damage_type']]
                    total_dmg = weighted_dmg_area / total_sign_area

                    # Calculate quadrant-wise damage
                    sign_mask_binary = cv2.threshold(cv2.cvtColor(sign_mask_cv, cv2.COLOR_BGR2GRAY),
                                                    20, 255, cv2.THRESH_BINARY)[1]
                    dmg_mask_quads = divide_sign_mask_quadrants(sign_mask_binary, f"{mask_id}.png", debug=False, save=False)
                    quad_dmgs = []
                    for quad_mask in dmg_mask_quads:
                        quad = cv2.bitwise_and(sign_mask_cv, sign_mask_cv, mask=quad_mask)
                        ##
                        # cv2.imshow('quad', quad)
                        # cv2.waitKey(0)
                        ##
                        props = get_mask_properties(quad)
                        total_quad_sign_area = props[4]
                        weighted_quad_dmg_area = 0
                        for dmg in all_masks[image['ID']][mask_id]['damages']:
                            dmg_mask = cv2.cvtColor(np.array(dmg['mask']), cv2.COLOR_RGB2GRAY)
                            dmg_mask = cv2.bitwise_and(dmg_mask, dmg_mask, mask=quad_mask)
                            props = get_mask_properties(dmg_mask)
                            if props is None:
                                quad_dmg_mask_area = 0  # The selected damage isn't in this quadrant
                            else:
                                quad_dmg_mask_area = props[4]
                            weighted_quad_dmg_area += quad_dmg_mask_area * dmg_weights[dmg['damage_type']]
                        quad_dmgs.append(weighted_quad_dmg_area / total_quad_sign_area)
                    # print("quad dmgs:", quad_dmgs)  ##
                    # print("total dmg:", total_dmg)  ##

                    all_masks[image['ID']][mask_id]['combined_bbox'] = (min_miny, max_maxy, min_minx, max_maxx)
                    all_masks[image['ID']][mask_id]['total_dmg'] = weighted_dmg_area / total_sign_area
                    all_masks[image['ID']][mask_id]['quad_dmgs'] = quad_dmgs

                    shape = sign_mask_cv.shape
                    h, w = shape[0], shape[1]
                    if args.debug_viz:
                        vis_images[image['ID']][max(min_miny, 0), :] = (0, 255, 0)
                        vis_images[image['ID']][min(max_maxy, h-1), :] = (0, 255, 0)
                        vis_images[image['ID']][:, max(min_minx, 0)] = (0, 255, 0)
                        vis_images[image['ID']][:, min(max_maxx, w-1)] = (0, 255, 0)

                    # Write to COCO format
                    write_label_coco(
                        labels_dict,
                        curr_ann_id,
                        coco_img_id,
                        lbox_img_id,
                        sign_mask_cv.shape,
                        (min_minx, max_maxx, min_miny, max_maxy),
                        total_dmg,
                        quad_dmgs,
                        copy_axes=False
                    )
                    curr_ann_id += 1

    if args.debug_viz:
        for img_id in vis_images:
            Image.fromarray(vis_images[img_id].astype(np.uint8)).save(f"{out_dir}/{img_id}.png")

    # Finalise COCO labels
    labels_dict['images'] = sorted(labels_dict['images'], key=lambda x: x['id'])
    labels_dict['annotations'] = sorted(labels_dict['annotations'], key=lambda x: x['id'])
    json.dump(labels_dict, labels_file, indent=4)
    labels_file.close()

    # Create a .npy file to store ground truths, for more efficient evaluation  
    # Format [image_id, xtl, ytl, width, height, damage_1, damage_2, ..., damage_n, class_id]
    annotations_array = []
    for ann in labels_dict['annotations']:
        row = [ann['image_id'], ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]]
        row.extend(ann['sector_damage'])
        row.append(ann['category_id'])
        annotations_array.append(row)
    with open(os.path.join(out_dir, "_single_annotations_array.npy"), "wb") as f:
        np.save(f, annotations_array)

if __name__ == "__main__":
    main()
