"""
Script parses a list input of mcd or tiff files with a common panel and provides a JSON
output of auto-scaled intensity values on the upper bound
"""
from ccramic.utils.pixel_level_utils import (
    get_default_channel_upper_bound_by_percentile)
from readimc import MCDFile
from ccramic.utils.alert import PanelMismatchError
import sys
import argparse
import json
import warnings
from statistics import mean, median

def cli_parser():
    parser = argparse.ArgumentParser(add_help=False,
            description="Parse a list input of mcd or tiff files with a common panel and provides a JSON "
                        "output of auto-scaled intensity values on the upper bound",
            usage='Example:\n python autoscale_roi.py -i first-mcd second.mcd -o autoscale.json -m mean -v -ex "laser,test,Start,End"')
    parser.add_argument('-i', "--input", nargs='+',
                        help="Series of paths to mcd or tiff files",
                        dest="input", type=str, required=True)
    parser.add_argument('-h', "--help", action="help",
                        help="Show the help/options menu and exit. Does not execute the application.",
                        dest="help")
    parser.add_argument('-pr', "--percentile", action="store",
                        help="Set the percentile of pixel intensities to use for the upper bound",
                        dest="percentile", default=99, type=float)
    parser.add_argument('-m', "--method", action="store",
                        help="Method for selecting the auto scaled upper bound. Options are either `min`, "
                             "`mean`, or `median`",
                        dest="method", default="min", type=str, choices=['min', 'mean', 'median'])
    parser.add_argument('-o', "--outfile", action="store",
                        help="Set the output tiff file. Default is geojson.tiff written to the current directory",
                        dest="outfile", default="autoscale.json", type=str)
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="If using verbose, print the current ROI parsing to the console.",
                        dest="verbose", default=False)
    parser.add_argument('-ex', "--keywords-exclude", action="store",
                        help="Pass a string of comma separated keywords to identify ROIs to exclude.",
                        dest="exclude", default="", type=str)


    return parser

def autoscale_upper_bound_from_mode(vals: list, mode="min"):
    if mode == "median":
        return median(vals)
    elif mode == "mean":
        return mean(vals)
    return min(vals)


def main(sysargs = sys.argv[1:]):
    warnings.filterwarnings("ignore")
    parser = cli_parser()
    args = parser.parse_args(sysargs)
    keywords_exclude = args.exclude.split(',') if args.exclude else []
    channel_scales = {}
    aliases = {}
    for file in args.input:
        if file.endswith('.mcd'):
            with MCDFile(file) as mcd_file:
                for slide in mcd_file.slides:
                    for acq in slide.acquisitions:
                        if not any([ignore in acq.description for ignore in keywords_exclude]):
                            img = mcd_file.read_acquisition(acq, strict=False)
                            if args.verbose:
                                print(f"Parsing ROI: {acq.description}")
                            if channel_scales and len(channel_scales.keys()) != len(acq.channel_names):
                                raise PanelMismatchError(f'One or more ROIs read from {args.input} contains different '
                                                         f'panel lengths. This is currently not supported by ccramic or its'
                                                         f' associated parsers.')
                            for channel_name, channel_array, channel_label in zip(acq.channel_names, img,
                                                                                  acq.channel_labels):
                                if channel_name not in channel_scales:
                                    channel_scales[channel_name] = []
                                channel_scales[channel_name].append(
                                    get_default_channel_upper_bound_by_percentile(channel_array, args.percentile))
                                aliases[channel_name] = channel_label
    json_template = {"channels": {}, "config": {"blend": [], "filter": {"global_apply_filter":
            False, "global_filter_type": 'median',
            "global_filter_val": 3, "global_filter_sigma": 1}}, "cluster": None, "gating": None}
    for channel, autoscale in channel_scales.items():
        # TODO: include different modes for selecting the appropriate upper bound
        json_template["channels"][channel] = {"color": "#FFFFFF", "x_lower_bound": None,
                                "x_upper_bound": autoscale_upper_bound_from_mode(autoscale, args.method),
                                "filter_type": None, "filter_val": None, "filter_sigma": None, "alias": aliases[channel]}
    with open(args.outfile, 'w', encoding='utf-8') as f:
            json.dump(json_template, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
