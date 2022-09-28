"""Adapted from:
https://github.com/hugovk/random-street-view/blob/main/random_street_view.py
"""

import argparse
import json
import os
import random
import sys
import requests
from urllib.request import urlopen, urlretrieve
from datetime import datetime
import contextlib
import yaml

import shapefile  # pip install pyshp
import mapillary.interface as mly  # pip install mapillary

# Optional, http://stackoverflow.com/a/1557906/724176
try:
    import timing
    assert timing  # avoid flake8 warning
except ImportError:
    pass

IMG_SUFFIX = "jpg"
MAX_TRIES = 10  # Used to set number of maximum attempts at finding a non-filtered image


with open("api_key.yaml", "r") as ymlfile:
    key = yaml.load(ymlfile, Loader=yaml.FullLoader)
token = key['token']
mly.set_access_token(token)

parser = argparse.ArgumentParser(
    description="Get random Street View images from within the borders of a given country. http://bboxfinder.com may "
    "be helpful for creating box coordinates and https://www.mapillary.com/app may be helpful for checking those boxes "
    "contain any images. By default, images are filtered out if they have any traffic signs as detected by Mapillary's "
    "systems; this should not be trusted absolutely, images should be manually checked for signs.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("country", help="ISO 3166-1 Alpha-3 Country Code, 'none', or 'near_global'")
help_str = "Enter 0 to use the default value given by the country's borders. Min/max lon/lat ignored if 'near_global'."
parser.add_argument("min_lon", type=float, help=help_str)
parser.add_argument("min_lat", type=float, help=help_str)
parser.add_argument("max_lon", type=float, help=help_str)
parser.add_argument("max_lat", type=float, help=help_str)
parser.add_argument("-n", "--images-wanted", type=int, default=10, help="Total number of images wanted.")
parser.add_argument("-b", "--burst", type=int, default=10, help="The maximum number of nearby images downloaded from "
                    "any random geographical point that hits. >1 (e.g. 10) is recommended if using 'near_global'. Note "
                    "that nearby images may be captured by the same camera on the same day, so there is a trade-off "
                    "between speed of image retrieval and diversity of images to chosen here. I have so far been "
                    "unable to determine the definition of 'nearby' from the Mapillary SDK documentation, but each "
                    "point hit will often return 100s-1000s of nearby images as shown by '--save-to-json'.")
parser.add_argument("-j", "--save-to-json", action="store_true", help="Save to a JSON file metadata of images found "
                    "near a point.")
parser.add_argument("-N", "--no-filter", action="store_true", help="Turn off filtering of images with traffic signs.")
args = parser.parse_args()

# TODO: "--all-in-box", '-A' mode where every single image in the box is downloaded, not just random ones; ADD A WARNING


# Determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
# http://www.ariel.com.au/a/python-point-int-poly.html
def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


print("Loading borders...")
shape_file = "TM_WORLD_BORDERS-0.3.shp"
if not os.path.exists(shape_file):
    sys.exit(
        f"Cannot find {shape_file}. Please download it from "
        "http://thematicmapping.org/downloads/world_borders.php and try again."
    )

sf = shapefile.Reader(shape_file, encoding="latin1")
shapes = sf.shapes()

if args.country.lower() == "none":
    if args.min_lon == 0 or args.min_lat == 0 or args.max_lon == 0 or args.max_lat == 0:
        sys.exit("A valid bounding box must be entered if no country is specified.")
    min_lon = args.min_lon
    min_lat = args.min_lat
    max_lon = args.max_lon
    max_lat = args.max_lat
    borders = []
elif args.country.lower() == "near_global":
    min_lon = -160
    min_lat = -56
    max_lon = 180
    max_lat = 71
    borders = []
else:
    print("Finding country...")
    for i, record in enumerate(sf.records()):
        if record[2] == args.country.upper():
            print(record[2], record[4])
            print(shapes[i].bbox)
            min_lon = shapes[i].bbox[0] if args.min_lon == 0 else args.min_lon
            min_lat = shapes[i].bbox[1] if args.min_lat == 0 else args.min_lat
            max_lon = shapes[i].bbox[2] if args.max_lon == 0 else args.max_lon
            max_lat = shapes[i].bbox[3] if args.max_lat == 0 else args.max_lat
            borders = shapes[i].points
            break

print("Getting images...")
attempts, country_hits, point_hits, point_misses, imagery_hits, imagery_misses, imagery_filtered = 0, 0, 0, 0, 0, 0, 0

dtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
outdir = f"mly_random_{args.country.upper()}_{dtime}"

img_ids = []
try:
    while True:
        attempts += 1
        rand_lat = random.uniform(min_lat, max_lat)
        rand_lon = random.uniform(min_lon, max_lon)

        # Check if (lon, lat) is inside country borders
        point_inside = True
        if borders != []:
            point_inside = point_inside_polygon(rand_lon, rand_lat, borders)

        if point_inside:
            # print("In country")
            country_hits += 1

            try:
                # We will only retrieve flat images, although the MTSD contains 1138 flattened 360° panorama images
                # NOTE: I couldn't figure out how to suppress the GET prints from the Mapillary SDK
                print()
                images = mly.get_image_close_to(longitude=rand_lon, latitude=rand_lat, image_type='flat',
                                                fields=['thumb_original_url']).to_dict()
                point_hits += 1
            except IndexError:
                point_misses += 1
                print(f"  No images found close to point")
                continue

            hit_dir = f"hits_{args.country.upper()}_{dtime}"
            if args.save_to_json:
                os.makedirs(hit_dir, exist_ok=True)
                with open(os.path.join(hit_dir, f"hit_#{point_hits}_with_{len(images['features'])}_imgs_"
                                       f"{rand_lon}_{rand_lat}.json"), mode="w") as f:
                    json.dump(images, f, indent=4)

            print('\n\n  lat,lon: ' + str(rand_lat) + ',' + str(rand_lon))
            ii = 0
            found = 0
            while found < args.burst and ii < (found + 1) * MAX_TRIES:
                if ii >= len(images['features']):
                    break
                img_id = images['features'][ii]['properties']['id']
                if img_id in img_ids:
                    imagery_misses += 1
                    ii += 1
                    continue
                url_request = f"https://graph.mapillary.com/{img_id}?access_token={token}&fields=thumb_original_url"
                response = requests.get(url_request).json()  # Query the API for the original image URL
                try:
                    url = response['thumb_original_url']
                except KeyError:
                    print(f"  Error retrieving image URL for {img_id}")
                    imagery_misses += 1
                    ii += 1
                    continue

                # Filter out images with traffic signs detected by Mapillary
                passed_filter = True
                if not args.no_filter:
                    # NOTE: I couldn't figure out how to suppress the GET prints from the Mapillary SDK
                    detections = mly.get_detections_with_image_id(img_id).to_dict()
                    def is_sign(detection):
                        v = detection['properties']['value']
                        return (
                            "complementary" in v or
                            "information" in v or
                            "regulatory" in v or
                            "warning" in v
                        )
                    signs = [d['properties']['value'] for d in detections['features'] if not is_sign(d)]
                    if signs != []:
                        imagery_filtered += 1
                        passed_filter = False
                        print("  ----- Skipped image with traffic sign detections -----")

                if passed_filter:
                    os.makedirs(outdir, exist_ok=True)
                    outfile = os.path.join(outdir, f"{img_id}.{IMG_SUFFIX}")
                    try:
                        # Download the image
                        data = requests.get(url)
                        with open(outfile, "wb") as f:
                            f.write(data.content)
                    except KeyboardInterrupt:
                        sys.exit("exit")

                    if os.path.isfile(outfile):
                        print(f"  ========== Got one! Taken from this point: {found + 1} "
                              f"(from {ii + 1} attempts) ==========")
                        img_ids.append(img_id)
                        imagery_hits += 1
                        found += 1
                        if imagery_hits >= args.images_wanted:
                            break
                    else:
                        imagery_misses += 1
                ii += 1
            if imagery_hits >= args.images_wanted:
                break
        else:
            # print("  Point outside country")
            pass
except KeyboardInterrupt:
    print("Keyboard interrupt")

stats_str = f"Attempts:\t{attempts}\n"
if borders != []:
    stats_str += f"Country hits:\t{country_hits}\n"
stats_str += f"Point misses:\t{point_misses}\n"
stats_str += f"Point hits:\t{point_hits}\n"
stats_str += f"Imagery misses:\t{imagery_misses}\n"
if not args.no_filter:
    stats_str += f"Filtered out:\t{imagery_filtered}\n"
stats_str += f"Imagery hits:\t{imagery_hits}"
print(f"\n{stats_str}")
with open(os.path.join(outdir, "stats.txt"), mode="w") as f:
    f.write(stats_str)
