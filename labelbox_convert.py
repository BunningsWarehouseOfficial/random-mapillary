import yaml
import json
import argparse

from PIL import Image
import numpy as np
import labelbox
from labelbox.data.annotation_types import Geometry
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
    'fading_discoloring_0.5': 0.5,  # Perhaps two: 0.6666 and 0.3333?
    'dirt': 1.0,
    'obscured_dark_shadow_0.8': 0.0
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug-viz", action="store_true", help="Debug visualisation mode.")
args = parser.parse_args()


def get_image_label(label, colors):
    image_np = label.data.value

    # blank_image = Image.new('RGB', (image_np.shape[1], image_np.shape[0]), (0, 0, 0))

    # Draw the annotations onto the source image
    # for ii, annotation in enumerate(label.annotations):
    #     if isinstance(annotation.value, Geometry):
    #         image_np = annotation.value.draw(canvas=image_np,
    #                                          color=colors[annotation.name],
    #                                          thickness=5)

    #         ##
    #         # ann = annotation.value.draw(color=colors[annotation.name],
    #         #                               thickness=5)
    #         # Image.fromarray(ann.astype(np.uint8)).save(f"{ii}_{annotation.extra['feature_id']}_{label.uid}.png")
    #         ##

    #         ##
    #         blank_image = annotation.value.draw(canvas=(None if ii == 0 else blank_image),
    #                                             color=colors[annotation.name],
    #                                             thickness=5)
    #         ##
    # Image.fromarray(blank_image.astype(np.uint8)).save(f"blank_{annotation.extra['feature_id']}_{label.uid}.png")

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
        print(bbox)  ##
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
    p_id = key['project_id']
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

    all_masks = {}
    if args.debug_viz:
        vis_images = {}

    # Label: https://github.com/Labelbox/labelbox-python/blob/develop/labelbox/data/annotation_types/label.py
    for label in labels:
        # if label.uid != "cl81ana824ww1071xgtgp6tjy":
        #     continue
        if args.debug_viz:
            vis_images[label.uid] = get_image_label(label, colors)
        all_masks[label.uid] = image_masks_dict(label, colors)

    with open("labels.json", "r") as f:
        labels_json = json.load(f)
    for image in labels_json:
        # if image['ID'] != "cl81ana824ww1071xgtgp6tjy":
        #     continue

        # Link damage masks to sign masks
        relations = image['Label']['relationships']
        for relation in relations:
            target_sign = relation['data']['target']
            source_dmg = relation['data']['source']
            all_masks[image['ID']][target_sign]['damages'].append(all_masks[image['ID']][source_dmg])

        # Find bbox for each combined mask
        for mask_id in all_masks[image['ID']]:
            if all_masks[image['ID']][mask_id]['is_sign']:
                # Get bbox for combined mask and create combined mask itself
                sign_mask = all_masks[image['ID']][mask_id]['mask']
                sign_mask_cv = cv2.cvtColor(np.array(sign_mask), cv2.COLOR_RGB2BGR)
                cv2.imshow('sign_mask_cv', sign_mask_cv)
                cv2.waitKey(0)
                min_miny, max_maxy, min_minx, max_maxx, sign_mask_area, _, _ = get_mask_properties(sign_mask)
                total_sign_area = sign_mask_area
                weighted_dmg_area = 0
                for dmg in all_masks[image['ID']][mask_id]['damages']:
                    dmg_mask = cv2.cvtColor(np.array(dmg['mask']), cv2.COLOR_RGB2BGR)
                    sign_mask_cv = cv2.bitwise_or(sign_mask_cv, dmg_mask)
                    # cv2.imshow('dmg_mask_cv', dmg_mask)
                    cv2.imshow('sign_mask_cv', sign_mask_cv)
                    cv2.waitKey(0)

                    # Calculate total damage
                    dmg_mask_gray = cv2.cvtColor(dmg_mask, cv2.COLOR_BGR2GRAY)
                    miny, maxy, minx, maxx, dmg_mask_area, _, _ = get_mask_properties(dmg_mask_gray)
                    min_miny = min(miny, min_miny)
                    max_maxy = max(maxy, max_maxy)
                    min_minx = min(minx, min_minx)
                    max_maxx = max(maxx, max_maxx)
                    total_sign_area += dmg_mask_area
                    weighted_dmg_area += dmg_mask_area * dmg_weights[dmg['damage_type']]

                # Calculate quadrant-wise damage
                sign_mask_binary = cv2.threshold(cv2.cvtColor(sign_mask_cv, cv2.COLOR_BGR2GRAY),
                                                 20, 255, cv2.THRESH_BINARY)[1]
                dmg_mask_quads = divide_sign_mask_quadrants(sign_mask_binary, f"{mask_id}.png", debug=True, save=True)
                quad_dmgs = []
                for quad_mask in dmg_mask_quads:
                    quad = cv2.bitwise_and(sign_mask_cv, sign_mask_cv, mask=quad_mask)
                    cv2.imshow('quad', quad)
                    cv2.waitKey(0)
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
                print("quad dmgs:", quad_dmgs)  ##
                print("total dmg:", weighted_dmg_area / total_sign_area)  ##

                all_masks[image['ID']][mask_id]['combined_bbox'] = (min_miny, max_maxy, min_minx, max_maxx)
                all_masks[image['ID']][mask_id]['total_dmg'] = weighted_dmg_area / total_sign_area
                all_masks[image['ID']][mask_id]['quad_dmgs'] = quad_dmgs

                if args.debug_viz:
                    vis_images[image['ID']][min_miny, :] = (0, 255, 0)
                    vis_images[image['ID']][max_maxy, :] = (0, 255, 0)
                    vis_images[image['ID']][:, min_minx] = (0, 255, 0)
                    vis_images[image['ID']][:, max_maxx] = (0, 255, 0)

    if args.debug_viz:
        for img_id in vis_images:
            Image.fromarray(vis_images[img_id].astype(np.uint8)).save(f"{img_id}.png")


if __name__ == "__main__":
    main()


# TODO: Get quadrant-wise damage (try out regionprops things)

# TODO: Option to exclude obscured_dark_shadow_0.8 and obscured_physical

# TODO: Convert to COCO annotations
