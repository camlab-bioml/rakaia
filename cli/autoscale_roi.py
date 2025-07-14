"""
Script parses a list input of mcd or tiff files with a common panel and provides a JSON
output of auto-scaled intensity values on the upper bound.
Parses the channel arrays for every ROI in the dataset, and appends a random subset of pixels to
an array to compute the percentile thresholds.
"""
import sys
from typing import Union
from itertools import chain
from pathlib import Path
import argparse
import json
import warnings
import glob
import numpy as np
from readimc import MCDFile, TXTFile

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

def iterate_rois(import_file: Union[str, Path],
                 keywords_exclude: list,
                 size_limit: int):
    """
    Iterate a single ROI from .txt or a series of ROIs from .mcd by parsing the file extension
    Return the reader object for each file type, as well as either a list of acquisitions for mcd,
    or a single numpy array for .txt
    Additionally, return the lists of channel names (keys), labels (values), and
    roi identifiers (acquisition description from .mcd, or file basename for .txt)
    """
    rois = []
    acq_channel_names = []
    acq_channel_labels = []
    roi_identifiers = []
    acq_reader = None
    if str(import_file).endswith('.mcd'):
        acq_reader = MCDFile(import_file)
        with acq_reader as mcd_open:
            for slide in mcd_open.slides:
                for acq in slide.acquisitions:
                    if not any([ignore in acq.description for ignore in keywords_exclude]) and \
                            (acq.height_px > size_limit and acq.width_px > size_limit):
                        rois.append(acq)
                        roi_identifiers.append(str(acq.description))
                        if not acq_channel_names and not acq_channel_labels:
                            acq_channel_names = acq.channel_names
                            acq_channel_labels = acq.channel_labels
    elif str(import_file).endswith('.txt'):
        acq_reader = TXTFile(import_file)
        with acq_reader as txt_open:
            # rois = txt_open.read_acquisition()
            rois = None
            roi_identifiers = str(Path(import_file).stem)
        if not acq_channel_names and not acq_channel_labels:
            acq_channel_names = txt_open.channel_names
            acq_channel_labels = txt_open.channel_labels
    return acq_reader, rois, acq_channel_names, acq_channel_labels, roi_identifiers

def append_roi_channel_subsets(subset_dict: dict, aliases_dict: dict,
                               acq_img: np.array, acq_channel_names: list,
                               acq_channel_labels: list,
                               proportion_to_use: float=0.01,
                               proportional_subsampling: bool=False,
                               subsample_size: int=5000):
    """
    Iterate an ROI and append a subset of pixels for every channel
    """
    subset_dict = {} if not subset_dict else subset_dict
    for channel_name, channel_array, channel_label in zip(acq_channel_names, acq_img,
                                                          acq_channel_labels):
        if channel_name not in subset_dict:
            subset_dict[channel_name] = np.empty((0,), dtype=float)
        array_use = channel_array.flatten().astype(np.uint32)
        # array_use = array_use[array_use > 0]
        subset_num = int(float(proportion_to_use) * int(array_use.shape[0])) if \
            proportional_subsampling else subsample_size
        flat_indices = np.random.choice(int(array_use.shape[0]),
                                        subset_num, replace=False)
        subset = array_use[flat_indices]
        subset_dict[channel_name] = np.concatenate(
            (subset_dict[channel_name], subset))

        if channel_name not in aliases_dict:
            aliases_dict[channel_name] = channel_label
    return subset_dict, aliases_dict

def main(sysargs=sys.argv[1:]):
    warnings.filterwarnings("ignore")
    parser = cli_parser()
    args = parser.parse_args(sysargs)
    keywords_exclude = args.exclude.split(',') if args.exclude else []
    channel_scales = {}
    aliases = {}
    # toggle proportional subsampling for percentage of ROI pixels, or use flat number for every ROI
    # if the files pass the initial fileparser, then can proceed
    input_paths = args.input if isinstance(args.input, list) else [args.input]
    files_to_process = []
    for path in input_paths:
        files_to_process += list(chain(glob.glob(f"{path}/*.mcd"), glob.glob(f"{path}/*.txt")))
    for file in files_to_process:
        reader, roi_to_read, channel_names, channel_labels, identifiers = iterate_rois(
            file, keywords_exclude, args.size_limit)
        if roi_to_read and isinstance(roi_to_read, list):
            with reader as file_open:
                for roi, acq_name in zip(roi_to_read, identifiers):
                    if args.verbose:
                        print('\033[32m' + f"Parsing ROI: {acq_name}")
                    img = file_open.read_acquisition(roi)
                    channel_scales, aliases = append_roi_channel_subsets(channel_scales,
                                            aliases, img, channel_names, channel_labels)
        else:
            if args.verbose:
                print('\033[32m' + f"Parsing ROI: {identifiers}")
                with reader as file_open:
                    roi_to_read = file_open.read_acquisition()
                    channel_scales, aliases = append_roi_channel_subsets(channel_scales, aliases, roi_to_read,
                                                          channel_names, channel_labels)
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
