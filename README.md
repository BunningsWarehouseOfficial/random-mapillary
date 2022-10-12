# Random Mapillary Images

Command-line Python 3 script to get images from random locations within specific countries using the Mapillary API. Intended to source backgrounds for semi-synthetic traffic sign detection training data.

Random (latitude, longitude) coordinates are generated from the country's border's bounding box, then checked to make sure they're within the actual borders. The desired number of corresponding Mapillary images near that coordinate point, as specified by the `-b` flag, are then downloaded. Images are ignored if the Mapillary back-end detects any traffic signs, but it often misses signs so don't rely on this. Repeat until the required number of images have been fetched.

A custom bounding box may also be used in combination with requiring presence within a country's borders. Alternatively, if the country is specified as 'none' then the custom bounding box will be the only restriction. This can, for example, be used to broadly, but not precisely, retrieve images from only a specific region or city.

http://bboxfinder.com may be helpful for creating box coordinates and https://www.mapillary.com/app may be helpful for checking that those boxes contain any images.

## Acknowledgement
This is a direct modification of Hugo van Kemenade's [random street view](https://github.com/hugovk/random-street-view) repository. \
All credit for the original code goes to them.

## Prerequisites

- Python 3.7+
- Install the Mapillary Python SDK and the Python Shapefile Library:
  `python -m pip install "mapillary==1.0.3" "pyshp>=2"` or \
  `pip install -r requirements.txt`
- Get the World Borders Dataset (e.g. TM_WORLD_BORDERS-0.3.shp) from
  http://thematicmapping.org/downloads/world_borders.php
- Request a free Mapillary client API key from https://www.mapillary.com/developer and place it into an `api_key.yaml` file formatted the same way as `api_key.yaml.example`.

## Usage
```
usage: random_mapillary.py [-h] [-n IMAGES_WANTED] [-b BURST] [-j] [-N] country min_lon min_lat max_lon max_lat

Get random Street View images from within the borders of a given country. http://bboxfinder.com may be helpful
for creating box coordinates and https://www.mapillary.com/app may be helpful for checking those boxes contain
any images. By default, images are filtered out if they have any traffic signs as detected by Mapillary's
systems; this should not be trusted absolutely, images should be manually checked for signs.

positional arguments:
  country               ISO 3166-1 Alpha-3 Country Code, 'none', or 'near_global'
  min_lon               For default from country borders, enter 0. Ignored if 'near_global'.
  min_lat               For default from country borders, enter 0. Ignored if 'near_global'.
  max_lon               For default from country borders, enter 0. Ignored if 'near_global'.
  max_lat               For default from country borders, enter 0. Ignored if 'near_global'.

optional arguments:
  -h, --help            show this help message and exit
  -n IMAGES_WANTED, --images-wanted IMAGES_WANTED
                        Total number of images wanted. (default: 100)
  -b BURST, --burst BURST
                        The maximum number of nearby images downloaded from any random geographical point that
                        hits. >1 (e.g. 10) is recommended if using 'near_global'. Note that nearby images may be
                        captured by the same camera on the same day, so there is a trade-off between speed of
                        image retrieval and diversity of images to chosen here. I have so far been unable to
                        determine the definition of 'nearby' from the Mapillary SDK documentation, but each
                        point hit will often return 100s-1000s of nearby images as shown by '--save-to-json'.
                        (default: 10)
  -j, --save-to-json    Save to a JSON file metadata of images found near a point. (default: False)
  -N, --no-filter       Turn off filtering of images with traffic signs. (default: False)
```

### Examples
Default number of images *within Australia's borders*.
```sh
python pull_images_2.py AUS 0 0 0 0
```
Large set of *global* images from across most of the Earth's surface. Due to the random methodology and sparse distribution of images globally this will likely *take many hours to execute*.
```sh
python pull_images.py near_global 0 0 0 0 -n 10000 -b 10
```
Images from roughly the *Perth, Western Australia metropolitan area*:
```sh
python pull_images_2.py none 115.629601 -32.599250 116.151460 -31.542966
```
Save the *unfiltered metadata for all images* nearby a point in the United Kingdom.
```sh
python pull_images.py GBR 0 0 0 0 -n 1 -b 1 --save-to-json --no-filter
```