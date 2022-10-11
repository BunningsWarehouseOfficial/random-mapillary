import yaml
import labelbox
from labelbox.data.annotation_types import Geometry
from PIL import Image
import numpy as np
from skimage.measure import regionprops


# TODO: Merge damage masks into sign masks for boxes using damage relations
#       - Make dict of feature_id: sign_mask for all annotations that don't have damages relation (i.e. sign masks)
#       - Merge damage masks into sign mask in corresponding dict value for the key taken from outgoing side of damages relation
# TODO: Different damage weighting for each class of damage

def get_image_label(label, colours):
    image_np = label.data.value

    # Draw the annotations onto the source image
    ii = 0  ##
    for annotation in label.annotations:
        if isinstance(annotation.value, Geometry):
            image_np = annotation.value.draw(canvas=image_np,
                                             color=colours[annotation.name],
                                             thickness=5)
            
            # NOTE: blank[y, x] starts from 0,0 at top left 

            ##
            blank = annotation.value.draw(color=colours[annotation.name],
                                          thickness=5)
            
            regions = regionprops(blank)
            for props in regions:
                y0, x0, _ = props.centroid
                bbox = props.bbox
                miny, minx = bbox[0:2]
                maxy, maxx = bbox[3:5]
                blank[miny, :] = (0, 255, 0)
                blank[maxy, :] = (0, 255, 0)
                blank[:, minx] = (0, 255, 0)
                blank[:, maxx] = (0, 255, 0)

            Image.fromarray(blank.astype(np.uint8)).save(f"{ii}_{label.uid}.png")
            ii += 1
            ##
    return Image.fromarray(image_np.astype(np.uint8))

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

    # Label: https://github.com/Labelbox/labelbox-python/blob/develop/labelbox/data/annotation_types/label.py
    for label in labels:
        if label.uid == "cl8lksgsx551l0729a8m831za":  ##
            image = get_image_label(label, colours)
            image.save(f"{label.uid}.png")
            break  ##


if __name__ == "__main__":
    main()
