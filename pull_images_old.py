import json
import requests
import urllib.request
import webbrowser
import random
import os
from datetime import datetime
import sys
import argparse

import mapillary.interface as mly
import shapefile  # pip install "pyshp>=2"

# TODO: Make sure the API token isn't included in any Git repos; could read from a yaml config or api.key file
TOKEN = "INSERT_YOUR_API_KEY_HERE"  # Mapillary client API token
mly.set_access_token(TOKEN)

NUM_IMGS = 3  # Number of images to download
GLOBAL = {
    'west': -180,
    'south': -90,
    'east': 180,
    'north': 90
}


parser = argparse.ArgumentParser(
    description="Get random Mapillary images from a given country"
)
parser.add_argument("country", help="ISO 3166-1 Alpha-3 Country Code or 'global'")
parser.add_argument("-n", "--images", type=int, default=10,
                    help="Number of images wanted")
args = parser.parse_args()


## Source: https://github.com/hugovk/random-street-view/blob/main/random_street_view.py
print("Loading borders")
shape_file = "TM_WORLD_BORDERS-0.3.shp"
if not os.path.exists(shape_file):
    sys.exit(
        f"Cannot find {shape_file}. Please download it from "
        "http://thematicmapping.org/downloads/world_borders.php and try again."
    )
sf = shapefile.Reader(shape_file, encoding="latin1")
shapes = sf.shapes()

if args.country.lower() != 'global':
    print("Finding country")
    for i, record in enumerate(sf.records()):
        if record[2] == args.country.upper():
            print(record[2], record[4])
            print(shapes[i].bbox)
            min_lon = shapes[i].bbox[0]
            min_lat = shapes[i].bbox[1]
            max_lon = shapes[i].bbox[2]
            max_lat = shapes[i].bbox[3]
            borders = shapes[i].points
            break
## End of code from source
else:
    min_lon = GLOBAL['west']
    min_lat = GLOBAL['south']
    max_lon = GLOBAL['east']
    max_lat = GLOBAL['north']

box = {
    'west': min_lon,
    'south': min_lat,
    'east': max_lon,
    'north': max_lat
}
images = mly.images_in_bbox(box, image_type='flat').to_dict()
with open("images_in_bbox_global.json", mode='w') as f:  # Save the data as JSON
    json.dump(images, f, indent=4)

os.mkdir(f"mly_random_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")  # Create a directory to store the images
for ii in range(NUM_IMGS):
    # image = random.choice(mly.images_in_bbox(8.5, 49.0, 8.6, 49.1))  # Get a random image from the bounding box
    lng = random.uniform(-180.0, 180.0)
    lat = random.uniform(-90.0, 90.0)
    # lng = 47.435164
    # lat = 15.756867
    # print(lng, lat)
    # We will only retrieve flat images, although the MTSD contains 1138 flattened 360° panorama images
    images = mly.get_image_close_to(longitude=lng, latitude=lat, image_type='flat').to_dict()
    ##
    with open("get_image_close_to.json", mode='w') as f:  # Save the data as JSON
        json.dump(images, f, indent=4)
    ##
    # TODO: Check for Mapillary traffic sign detections to filter out those images, pick first with no detections
    img_id = images['features'][0]['properties']['id']
    url_request = f"https://graph.mapillary.com/{img_id}?access_token={TOKEN}&fields=thumb_original_url"
    response = requests.get(url_request).json()  # Query the API
    url = response['thumb_original_url']
    urllib.request.urlretrieve(url, f"mly_images/{img_id}.jpg")

# NOTE: There is a traffic_signs_in_bbox() function


# image = '236659458656576' #'916862465836208' #'745860926087097' #'1933525276802129'
# url_request = f"https://graph.mapillary.com/{image}?access_token={token}&fields=thumb_original_url"

# response = requests.get(url_request).json()  # Query the API
# if not 'error' in response.keys():
#     url = response['thumb_original_url']  # Retrieve the full resolution image URL

#     webbrowser.register("vivaldi",  # Open image in browser
#         None,
#         webbrowser.BackgroundBrowser("C://Users//Kristian//AppData//Local//Vivaldi//Application//vivaldi.exe"))
#     webbrowser.get('vivaldi').open(url)
#     print(f"\n{json.dumps(response, indent=4)}")

#     det_request = f"https://graph.mapillary.com/{image}/detections?access_token={token}&fields=image,value,geometry"
#     print(f"\n{json.dumps(requests.get(det_request).json(), indent=4)}")

#     # TODO: Check for Mapillary traffic sign detections to filter out those images
#     #       Filter out "regulatory--...", "warning--...", 

#     # TODO: Use EfficientDet-D7x to filter out images with traffic signs
# else:
#     print(response['error'])