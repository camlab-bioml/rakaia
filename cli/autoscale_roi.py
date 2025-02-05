"""
Define the scaling values required for rakaia
"""

"""
Script parses a list input of mcd or tiff files with a common panel and provides a JSON
output of auto-scaled intensity values on the upper bound.
Parses the channel arrays for every ROI in the dataset, and appends a random subset of pixels to
an array to compute the percentile thresholds.
"""
import sys
import argparse
import json
import warnings
import glob
import numpy as np
from readimc import MCDFile

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Parse a list input of mcd or tiff files with a common panel and provides a JSON "
                        "output of auto-scaled intensity values on the upper bound",
            usage='Example:\n python autoscale_roi.py -i directory_of_mcd_files -o autoscale.json -v -ex "laser,test,Start,End"')
    parser.add_argument('-i', "--input", nargs='+',
                        help="Series of paths to mcd or tiff files",
                        dest="input", type=str, required=True)
    parser.add_argument('-h', "--help", action="help",
                        help="Show the help/options menu and exit. Does not execute the application.",
                        dest="help")
    parser.add_argument('-pr', "--percentile", action="store",
                        help="Set the percentile of pixel intensities to use for the upper bound",
                        dest="percentile", default=99, type=float)
    parser.add_argument('-o', "--outfile", action="store",
                        help="Set the output tiff file. Default is geojson.tiff written to the current directory",
                        dest="outfile", default="autoscale.json", type=str)
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="If using verbose, print the current ROI parsing to the console.",
                        dest="verbose", default=False)
    parser.add_argument('-ex', "--keywords-exclude", action="store",
                        help="Pass a string of comma separated keywords to identify ROIs to exclude.",
                        dest="exclude", default="", type=str)
    parser.add_argument('-s', "--size", action="store",
                        help="Integer dimension threshold. ROIs with an x or y dimension below this value are not "
                             "considered. Default is 100 pixels",
                        dest="size_limit", default=100, type=int)
    parser.add_argument('-ss', "--subsample-size", action="store",
                        help="Integer for the number of pixels to subsample per array to compute the percentile.",
                        dest="subsample_size", default=5000, type=int)

    return parser

def main(sysargs=sys.argv[1:]):
    warnings.filterwarnings("ignore")
    parser = cli_parser()
    args = parser.parse_args(sysargs)
    keywords_exclude = args.exclude.split(',') if args.exclude else []
    channel_scales = {}
    aliases = {}
    # toggle proportional subsampling for percentage of ROI pixels, or use flat number for every ROI
    proportional_subsampling = False
    # if the files pass the initial fileparser, then can proceed
    input_paths = args.input if isinstance(args.input, list) else [args.input]
    mcds_to_process = []
    for path in input_paths:
        mcds_to_process += list(glob.glob(f"{path}/*.mcd"))
    for file in mcds_to_process:
        if file.endswith('.mcd'):
            with (MCDFile(file) as mcd_file):
                for slide in mcd_file.slides:
                    for acq in slide.acquisitions:
                        if not any([ignore in acq.description for ignore in keywords_exclude]) and \
                                (acq.height_px > args.size_limit and acq.width_px > args.size_limit):
                            img = mcd_file.read_acquisition(acq, strict=False)
                            if args.verbose:
                                print('\033[32m' + f"Parsing ROI: {acq.description}")
                            for channel_name, channel_array, channel_label in zip(acq.channel_names, img,
                                                                                  acq.channel_labels):
                                if channel_name not in channel_scales:
                                    channel_scales[channel_name] = np.empty((0,), dtype=float)
                                array_use = channel_array.flatten().astype(np.uint32)
                                # array_use = array_use[array_use > 0]
                                subset_num = int(0.01 * int(array_use.shape[0])) if \
                                    proportional_subsampling else args.subsample_size
                                flat_indices = np.random.choice(int(array_use.shape[0]),
                                                                subset_num, replace=False)
                                subset = array_use[flat_indices]
                                channel_scales[channel_name] = np.concatenate(
                                    (channel_scales[channel_name], subset))

                                aliases[channel_name] = channel_label
    json_template = {"channels": {}, "config": {"blend": [], "filter": {"global_apply_filter":[],
                    "global_filter_type": 'median', "global_filter_val": 3, "global_filter_sigma": 1}},
                    "cluster": None, "gating": None}
    for channel, arr in channel_scales.items():
        json_template["channels"][channel] = {"color": "#FFFFFF", "x_lower_bound": 0.0,
                                              "x_upper_bound": float(np.percentile(arr, args.percentile)),
                                              "filter_type": None, "filter_val": None, "filter_sigma": None,
                                              "alias": aliases[channel]}
    with open(args.outfile, 'w', encoding='utf-8') as f:
        json.dump(json_template, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
