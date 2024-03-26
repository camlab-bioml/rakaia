import argparse
import sys
import geojson
import numpy as np
from shapely.geometry import Polygon
import rasterio.features
import tifffile
from pathlib import Path

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Convert a geojson tiff into a greyscale tiff mask array. Requires the user to set the "
                        "output dimensions of the tiff.",
            usage='python geojson_to_tiff.py -i input.geojson -o output.tiff -x 1500 -y 1200')
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

    return parser

def main(sysargs = sys.argv[1:]):
    parser = cli_parser()
    args = parser.parse_args(sysargs)
    out_shape = (args.height, args.width)
    merged_mask = np.zeros(out_shape, dtype=np.float32)
    with open(Path(args.input)) as f:
        gj = geojson.load(f)
        feature_index = 1
        for feature in gj['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates']
                poly = Polygon(coords[0])
                img = rasterio.features.rasterize([poly], out_shape=out_shape)
                # TODO: still cannot handle combination of both cases:
                # Region 1 is largest, region 2 is smallest, and region 3 is medium, all with overlap

                # Case 1: subsequent region contains an entire existing region, and overwrites the entire one when added
                # Case 2: subsequent region is contained entirely inside existing region, and is not added because
                # regions already exist there

                # Case 1: incoming region is completely inside existing region
                # if the sum is 0, set the current mask overlap to 0
                incoming_inside = img[(img > 0) & (merged_mask == 0)]
                if int(np.sum(incoming_inside)) == 0:
                    merged_mask = np.where((img > 0) & (merged_mask > 0), 0, merged_mask)
                else:
                    # Case 2: overlap in existing
                    # set incoming overlap to 0
                    is_overlap = img[(merged_mask > 0) & (img > 0)]
                    if int(np.sum(is_overlap)) > 0:
                        img = np.where((img > 0) & (merged_mask > 0), 0, img)

                merged_mask = np.where(img > 0, feature_index, merged_mask)
                feature_index += 1

        tifffile.imwrite(args.outfile, merged_mask.astype(np.float32))

if __name__ == "__main__":
    main()
