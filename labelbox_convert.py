import yaml
import json
import argparse

from PIL import Image
import numpy as np
import labelbox
from labelbox.data.annotation_types import Geometry
from skimage.measure import regionprops


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug-viz", action="store_true", help="Debug visualisation mode.")
args = parser.parse_args()


# TODO: Different damage weighting for each class of damage

def get_image_label(label, colours):
    image_np = label.data.value

    # Draw the annotations onto the source image
    for annotation in label.annotations:
        if isinstance(annotation.value, Geometry):
            image_np = annotation.value.draw(canvas=image_np,
                                             color=colours[annotation.name],
                                             thickness=5)
    return image_np

def image_masks_dict(label, colours):
    image_masks = {}
    for annotation in label.annotations:
        is_sign = annotation.name == "Traffic Sign Mask"
        if isinstance(annotation.value, Geometry):
            image_masks[annotation.extra['feature_id']] = {
                'mask': annotation.value.draw(color=colours[annotation.name], thickness=5),
                'is_sign': True,
                'damages': [],
                'combined_bbox': None
            } if is_sign else {
                'mask': annotation.value.draw(color=colours[annotation.name], thickness=5),
                'is_sign': False,
                'damage_type': annotation.classifications[0].value.answer.name
            }
    return image_masks

def get_mask_bbox(mask):
    regions = regionprops(mask)
    for props in regions:
        cy, cx, _ = props.centroid
        bbox = props.bbox
        miny, minx = bbox[0:2]
        maxy, maxx = bbox[3:5]
        ##
        # mask[miny, :] = (0, 255, 0)
        # mask[maxy, :] = (0, 255, 0)
        # mask[:, minx] = (0, 255, 0)
        # mask[:, maxx] = (0, 255, 0)
        ##
        return miny, maxy, minx, maxx, cy, cx

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
    hex_to_rgb = lambda hex_color: tuple(
        int(hex_color[i + 1:i + 3], 16) for i in (0, 2, 4))
    colours = {
        tool.name: hex_to_rgb(tool.color)
        for tool in labelbox.OntologyBuilder.from_project(project).tools
    }

    all_masks = {}
    if args.debug_viz:
        vis_images = {}

    # Label: https://github.com/Labelbox/labelbox-python/blob/develop/labelbox/data/annotation_types/label.py
    for label in labels:
        if args.debug_viz:
            vis_images[label.uid] = get_image_label(label, colours)
        all_masks[label.uid] = image_masks_dict(label, colours)

    with open("labels.json", "r") as f:
        labels_json = json.load(f)
    for image in labels_json:
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
                min_miny, max_maxy, min_minx, max_maxx, _, _ = get_mask_bbox(sign_mask)
                for dmg in all_masks[image['ID']][mask_id]['damages']:
                    miny, maxy, minx, maxx, _, _ = get_mask_bbox(dmg['mask'])
                    min_miny = min(miny, min_miny)
                    max_maxy = max(maxy, max_maxy)
                    min_minx = min(minx, min_minx)
                    max_maxx = max(maxx, max_maxx)
                all_masks[image['ID']][mask_id]['combined_bbox'] = (min_miny, max_maxy, min_minx, max_maxx)

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
