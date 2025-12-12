"""
Use a headless version of rakaia file parsing and additive image blending to generate
a series of blended tiffs from a list of filepaths (CLI) using an existing JSON blend parameter
"""

import argparse
import os
import json
import warnings
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tifffile import imwrite
from rakaia.parsers.pixel import FileParser
from rakaia.parsers.lazy_load import SingleMarkerLazyLoader
from rakaia.utils.pixel import (
    split_string_at_pattern,
    get_additive_image,
    apply_preset_to_array,
    apply_filter_to_array,
    recolour_greyscale)

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Parse a series of files and apply additive blending with a headless version of rakaia, "
                        "outputting each file as a tiff. ",
            usage='Example:\n python apply_blend.py -i first.mcd second.mcd -o out_images/ -p param.json -t tiff')
    parser.add_argument('-i', "--input", nargs='+',
                        help="Series of paths to mcd, tiff, txt, or h5ad files",
                        dest="input", type=str, required=True)
    parser.add_argument('-o', "--outdir", action="store",
                        help="Set the output directory where blended tiffs will be written.",
                        dest="outdir", default="outdir", type=str, required=True)
    parser.add_argument('-p', "--params", action="store",
                        help="Path to a channel blend parameter JSON in rakaia-compatible output format.",
                        dest="params", default="param.json", required=True, type=argparse.FileType("r"))
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="If using verbose, print the current ROI parsing to the console.",
                        dest="verbose", default=False)
    parser.add_argument('-t', "--type", action="store",
                        help="Set the type of image output (either png or tiff). Default: tiff.",
                        dest="type", type=str, default="tiff")
    parser.add_argument('-h', "--help", action="help",
                        help="Show the help/options menu and exit. Does not execute the script.",
                        dest="help")

    return parser

def main(sysargs=sys.argv[1:]):
    warnings.filterwarnings("ignore")
    parser = cli_parser()
    args = parser.parse_args(sysargs)

    if args.type not in ['tiff', 'png']:
        print("\033[31m" + "ERROR: CLI argument `type` should be one of `tiff` or `png`. "
                           "Please review the CLI inputs.")
        sys.exit(1)

    files_to_process = args.input if isinstance(args.input, list) else [args.input]
    blend_params = json.load(args.params)

    os.makedirs(str(Path(args.outdir)), exist_ok=True)
    for file in files_to_process:
        # get the key representation of every dataset
        # use the initial params if .txt is used, as it is not lazy loaded
        use_lazy_load = False if str(file).endswith('.txt') else True
        initial_parse = {key: value for key, value in FileParser([str(file)],
                        lazy_load=use_lazy_load).image_dict.items() if key not in
                          ['metadata', 'metadata_columns']}
        for roi in list(initial_parse.keys()):
            try:
                array_loaded = initial_parse if not use_lazy_load else (
                    SingleMarkerLazyLoader(initial_parse, roi,
                                           {'uploads': files_to_process},
                                           blend_params['config']['blend']).get_image_dict())
                # use the acquisition id unless it's not unique
                exp, slide, acq = split_string_at_pattern(roi)
                # IMP: for mcds, include both the filename and the acquisition info to keep unique
                file_out_name = f"{exp}_{acq}" if acq != 'acq' else exp
                if args.verbose:
                    print('\033[32m' + f"Processing ROI: {file_out_name}")
                channel_thumbnails = {}
                for selected in blend_params['config']['blend']:
                    tile = apply_preset_to_array(array_loaded[roi][selected], blend_params['channels'][selected])
                    rgb_tile = recolour_greyscale(tile, blend_params['channels'][selected]['color'])
                    channel_thumbnails[selected] = rgb_tile
                additive_image = get_additive_image(channel_thumbnails, blend_params['config']['blend'])
                additive_image = apply_filter_to_array(additive_image,
                                                       blend_params['config']['filter']['global_apply_filter'],
                                                       blend_params['config']['filter']['global_filter_type'],
                                                       blend_params['config']['filter']['global_filter_val'],
                                                       blend_params['config']['filter']['global_filter_sigma'])
                additive_image = np.clip(additive_image, 0, 255)
                if args.type == 'tiff':
                    imwrite(str(os.path.join(args.outdir, f"{file_out_name}.tiff")),
                            additive_image.astype(np.uint8), photometric='rgb')
                else:
                    Image.fromarray(additive_image.astype(np.uint8)).convert('RGB').save(
                        str(os.path.join(args.outdir, f"{file_out_name}.png")), format="PNG")
            except (KeyError, ValueError):
                print("\033[31m" + "ERROR: The blend parameter JSON passed appears to be either not the expected "
                                   "rakaia JSON output format, or for a different panel than the files to be processed. "
                                   "Please review the CLI inputs.")
                sys.exit(1)
if __name__ == "__main__":
    main()
