# Supporting CLI parsers for rakaia

This directory contains additional CLI tools to
support rakaia that are not directly integrated into the
rakaia API.

**IMPORTANT**: `apply_blend.py` requires rakaia API access,
so installation from source is required before the script can be used.

# Scripts

## apply_blend.py

`apply_blend.py` takes a series of input files as well as a rakaia-generated
session JSON with a compatible panel, and creates a blended image
output in either `tiff` or `png` format. The script
accepts the same raw file extensions as rakaia for imaging files
(.mcd, .txt, .tiff, and .h5ad).

Options and usage:

```commandline
usage: Example:
python apply_blend.py -i first.mcd second.mcd -o out_images/ -p param.json -t tiff

Parse a series of files and apply additive blending with a headless version of rakaia, outputting each file as a tiff.

options:
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Series of paths to mcd, tiff, txt, or h5ad files
  -o OUTDIR, --outdir OUTDIR
                        Set the output directory where blended tiffs will be written.
  -p PARAMS, --params PARAMS
                        Path to a channel blend parameter JSON in rakaia-compatible output format.
  -v, --verbose         If using verbose, print the current ROI parsing to the console.
  -t TYPE, --type TYPE  Set the type of image output (either png or tiff). Default: tiff.
  -h, --help            Show the help/options menu and exit. Does not execute the script.
```

## autoscale_roi.py

`autoscale_blend.py` accepts either `.mcd` or `.txt` IMC files and performs
per-marker intensity thresholding based on all the ROIs passed.
Specifically, it will take a random subset of pixels from every
ROI and compute the 99th percentile upper bound for intensity adjustment,
thereby giving a more balanced visual scaling parameter in rakaia
to better differentiate positive and negative marker expression by region.

Example usage:

```commandline
usage: Example:
python autoscale_roi.py -i first-mcd second.mcd -o autoscale.json -m mean -v -ex "laser,test,Start,End"

Parse a list input of mcd or tiff files with a common panel and provides a JSON output of auto-scaled intensity values on the upper bound

options:
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Series of paths to mcd or tiff files
  -h, --help            Show the help/options menu and exit. Does not execute the application.
  -pr PERCENTILE, --percentile PERCENTILE
                        Set the percentile of pixel intensities to use for the upper bound
  -o OUTFILE, --outfile OUTFILE
                        Set the output tiff file. Default is geojson.tiff written to the current directory
  -v, --verbose         If using verbose, print the current ROI parsing to the console.
  -ex EXCLUDE, --keywords-exclude EXCLUDE
                        Pass a string of comma separated keywords to identify ROIs to exclude.
  -s SIZE_LIMIT, --size SIZE_LIMIT
                        Integer dimension threshold. ROIs with an x or y dimension below this value are not considered. Default is 100 pixels
  -ss SUBSAMPLE_SIZE, --subsample-size SUBSAMPLE_SIZE
                        Integer for the number of pixels to subsample per array to compute the percentile.
```

## geojson_to_tiff.py

`geojson_to_tiff.py` enables conversion of geoJSON files into a tiff mask,
where objects are encoded by integer IDs. \
**Important**: the script requires to user to set the tiff output
dimensions, as the geoJSON file does not capture this information.

The script also requires an installation of geopandas:

```commandline
pip install geopandas==1.1.1
```

Example usage:

```commandline
usage: Example:
python geojson_to_tiff.py -i input.geojson -o output.tiff -x 1500 -y 1200 -ht "Lumen"

Convert a geojson file into a greyscale tiff mask array. Requires the user to set the output dimensions of the tiff. Certain annotations can be ignored as holes/blank
in the final output mask.

options:
  -i INPUT, --input INPUT
                        Path input to a geoJSON file
  -h, --help            Show the help/options menu and exit. Does not execute the application.
  -x WIDTH, --width WIDTH
                        Set the width of the output tiff. Default: 1000 pixels
  -y HEIGHT, --height HEIGHT
                        Set the height of the output tiff. Default: 1000 pixels
  -o OUTFILE, --outfile OUTFILE
                        Set the output tiff file. Default is geojson.tiff written to the current directory
  -ht HOLE_TYPES, --hole-types HOLE_TYPES
                        A list of comma separated annotations to treat as holes (i.e. 0 in the final mask)
```
