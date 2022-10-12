import yaml
import json
import argparse

from PIL import Image
import numpy as np
import labelbox
from labelbox.data.annotation_types import Geometry
from skimage.measure import regionprops


damage_weights = {
    'graffiti': 1.0,
    'cracked': 1.0,
    'obscured_physical': 1.0,
    'missing_sections_holes': 1.0,
    'stickers': 1.0,
    'other_vandalism': 1.0,
    'rust_or_other_aging': 1.0,
    'chipped_paint': 1.0,
    'fading_discoloring_0.5': 0.5,
    'dirt': 1.0,
    'obscured_shadow_0.8': 0.8
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug-viz", action="store_true", help="Debug visualisation mode.")
args = parser.parse_args()


def get_image_label(label, colors):
    image_np = label.data.value

    # Draw the annotations onto the source image
    for ii, annotation in enumerate(label.annotations):
        if isinstance(annotation.value, Geometry):
            image_np = annotation.value.draw(canvas=image_np,
                                             color=colors[annotation.name],
                                             thickness=5)

            ##
            # blank = annotation.value.draw(color=colors[annotation.name],
            #                               thickness=5)
            # Image.fromarray(blank.astype(np.uint8)).save(f"{ii}_{annotation.extra['feature_id']}_{label.uid}.png")
            ##

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
                'total__dmg': None
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
        cy, cx, _ = props.centroid
        area = props.area  # Area of region pixels, not area of bounding box
        bbox = props.bbox
        miny, minx = bbox[0:2]
        maxy, maxx = bbox[3:5]
        ##
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
                # Get bbox for combined mask
                sign_mask = all_masks[image['ID']][mask_id]['mask']
                min_miny, max_maxy, min_minx, max_maxx, sign_mask_area, _, _ = get_mask_properties(sign_mask)
                total_sign_area = sign_mask_area
                weighted_dmg_area = 0
                for dmg in all_masks[image['ID']][mask_id]['damages']:
                    miny, maxy, minx, maxx, dmg_mask_area, _, _ = get_mask_properties(dmg['mask'])
                    min_miny = min(miny, min_miny)
                    max_maxy = max(maxy, max_maxy)
                    min_minx = min(minx, min_minx)
                    max_maxx = max(maxx, max_maxx)
                    total_sign_area += dmg_mask_area
                    weighted_dmg_area += dmg_mask_area * damage_weights[dmg['damage_type']]
                all_masks[image['ID']][mask_id]['combined_bbox'] = (min_miny, max_maxy, min_minx, max_maxx)
                all_masks[image['ID']][mask_id]['total_dmg'] = weighted_dmg_area / total_sign_area

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
