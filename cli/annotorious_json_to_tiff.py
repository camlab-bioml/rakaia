"""
Convert an `annotorious` WSI annotation JSON (custom exported from rakaia) into an
instance segmentation mask. each annotation will get a unique segmentation ID in the output mask.
Requires that the width and height be manually set by the user (not captured in the out JSON).
"""
import argparse
import sys
import numpy as np
import json
from typing import Union
import tifffile
from pathlib import Path
import warnings
from PIL import Image, ImageDraw

def load_annotations(json_path: Union[str, Path]):
    with open(json_path) as f:
        return json.load(f)

def polygons_from_annotations(polygon_data: list):
    """Extract polygon selectors from a list of Annotorious/W3C-style annotations."""
    polygons = []
    for ann in polygon_data:
        target = ann.get("target", {})
        selector = target.get("selector", {})
        if selector.get("type") != "POLYGON":
            continue
        geometry = selector.get("geometry", {})
        points = geometry.get("points", [])
        if len(points) < 3:
            continue
        polygons.append({
            "id": ann.get("id"),
            "points": points,
            "bounds": geometry.get("bounds"),
        })

    return polygons


def build_mask(polygons: list, width: int=1000, height: int=1000, start_label=1):
    """Draw each polygon filled with a unique integer label onto an int32 canvas."""
    canvas = Image.new("I", (width, height), 0)  # 32-bit signed int, safe for drawing
    draw = ImageDraw.Draw(canvas)

    for i, poly in enumerate(polygons, start=start_label):
        pts = [(x, y) for x, y in poly["points"]]
        draw.polygon(pts, fill=i)

    arr = np.array(canvas, dtype=np.int32)
    return arr

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Convert an annotorious WSI annotation JSON from rakaia into a greyscale tiff mask array. "
                        "Requires the user to set the output dimensions of the tiff.",
            usage='Example:\n python annotorious_json_to_tiff.py -i annotorious.json -o output.tiff -x 1500 -y 1200')
    parser.add_argument('-i', "--input", action="store",
                        help="Path input to an annotorious WSI JSON exported from rakaia",
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
                        help="Set the output tiff file. Default is annotorious_out.tiff written to the current directory",
                        dest="outfile", default="annotorious_out.tiff", type=str)

    return parser

def main(sysargs=sys.argv[1:]):
    warnings.filterwarnings("ignore")
    parser = cli_parser()
    args = parser.parse_args(sysargs)

    annotations = load_annotations(args.input)
    polygons = polygons_from_annotations(annotations)
    mask = build_mask(polygons, int(args.width), int(args.height))

    tifffile.imwrite(args.outfile, mask.astype(np.uint32), photometric='minisblack')

if __name__ == "__main__":
    main()
