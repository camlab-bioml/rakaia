import argparse
import sys
import numpy as np
import rasterio.features
import tifffile
from pathlib import Path
import geopandas
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import warnings

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Convert a geojson tiff into a greyscale tiff mask array. Requires the user to set the "
                        "output dimensions of the tiff. Certain annotations can be ignored as holes/blank in the "
                        "final output mask.",
            usage='Example:\n python geojson_to_tiff.py -i input.geojson -o output.tiff -x 1500 -y 1200 -ht "Lumen"')
    parser.add_argument('-i', "--input", action="store",
                        help="Path input to a geoJSON file",
                        dest="input", type=str, required=True)
    parser.add_argument('-h', "--help", action="help",
                        help="Show the help/options menu and exit. Does not execute the application.",
                        dest="help")
    parser.add_argument('-x', "--width", action="store",
                        help="Set the width of the output tiff. Default: 1000 pixels",
                        dest="width", default=1000, type=int)
    parser.add_argument('-y', "--height", action="store",
                        help="Set the height of the output tiff. Default: 1000 pixels",
                        dest="height", default=1000, type=int)
    parser.add_argument('-o', "--outfile", action="store",
                        help="Set the output tiff file. Default is geojson.tiff written to the current directory",
                        dest="outfile", default="geojson.tiff", type=str)
    parser.add_argument('-ht', "--hole-types", action="store",
                        help="A list of comma separated annotations to treat as holes (i.e. 0 in the final mask)",
                        dest="hole_types", default="", type=str)

    return parser

def main(sysargs=sys.argv[1:]):
    # TODO: what happens if a JSON has a blend of multi and single polygons? Is this possible?
    warnings.filterwarnings("ignore")
    # os.environ['USE_PYGEOS'] = '0'
    parser = cli_parser()
    args = parser.parse_args(sysargs)
    out_shape = (args.height, args.width)
    list_of_hole_types = args.hole_types.split(',') if args.hole_types else []
    merged_mask = np.zeros(out_shape, dtype=np.float32)
    georead = geopandas.read_file(Path(args.input))
    feature_index = 1
    for poly, annotation in zip(georead['geometry'], georead['classification']):
        if isinstance(poly, Polygon):
            # treat all individual polygons as separate objects, because it is difficult to tell which ones
            # may correspond to shapes vs. holes
            # converting a list of polygons to a multipolygon can cause loss of internal/nested shapes
            img = rasterio.features.rasterize([poly], out_shape=out_shape)
            # check to see if the annotation is in the list of holes. if so, set pixels in the output mask to 0
            if 'name' in annotation and annotation['name'] in list_of_hole_types:
                merged_mask = np.where(img > 0, 0, merged_mask)
            else:
                # if the annotation is kept as an object, check the conditions
                # Case 1: the incoming shape is contained entirely within an existing shape
                incoming_inside = img[(img > 0) & (merged_mask == 0)]
                if int(np.sum(incoming_inside)) == 0:
                    merged_mask = np.where((img > 0) & (merged_mask > 0), 0, merged_mask)
                else:
                    # Case 2: overlap in existing: set incoming overlap to 0
                    is_overlap = img[(merged_mask > 0) & (img > 0)]
                    if int(np.sum(is_overlap)) > 0:
                        img = np.where((img > 0) & (merged_mask > 0), 0, img)
                merged_mask = np.where(img > 0, feature_index, merged_mask)
                # only increment the feature index if it is not a hole
                feature_index += 1
        # if a multi polygon, the holes are already computed, so create an object array without modification
        elif isinstance(poly, MultiPolygon):
            img = rasterio.features.rasterize([poly], out_shape=out_shape)
            # TODO: decide if the cases above for single polygons should be checked for multi as well
            merged_mask = np.where(img > 0, img, merged_mask)

        tifffile.imwrite(args.outfile, merged_mask.astype(np.float32))


if __name__ == "__main__":
    main()
