import pandas as pd
import h5py
from pathlib import Path
from tifffile import TiffFile
import os
from ccramic.utils.pixel_level_utils import *
from ccramic.utils.cell_level_utils import validate_mask_shape_matches_image
from ccramic.utils.roi_utils import subset_mask_outline_using_cell_id_list
from ccramic.parsers.cell_level_parsers import match_mask_name_with_roi, match_mask_name_to_quantification_sheet_roi
from readimc import MCDFile, TXTFile
import random
import numpy as np
import cv2

def generate_multi_roi_images_from_query(session_config, blend_dict,
                                         currently_selected_channels, num_queries=5, rois_exclude=None,
                                         predefined_indices=None, mask_dict=None, dataset_options=None,
                                         query_cell_id_lists=None):
    """
    Generate a gallery of images for multiple ROIs using the current parameters of the current ROI
    Important: ignores the current ROI
    """
    # if the indices are predefined, do not cap
    # num_queries = num_queries if predefined_indices is None else len(predefined_indices)
    if predefined_indices is not None:
        query_selection = predefined_indices
    if rois_exclude is None:
        rois_exclude = []
    try:
        roi_images = {}
        # get the index of the file from the experiment number in the event that there are multiple uploads
        # file_path = None
        # for files_uploaded in session_config['uploads']:
        #     if str(Path(files_uploaded).stem) == basename:
        #         file_path = files_uploaded
        files = [file for file in session_config['uploads'] if str(file).endswith('.mcd')]
        if predefined_indices is None:
            random.shuffle(files)
        if files is not None and len(files) > 0:
            queries_obtained = 0
            for file_path in files:
                basename = str(Path(file_path).stem)
                with MCDFile(file_path) as mcd_file:
                    # generate a random number of roi indices to query
                    # query_length = min(len(dataset_options), num_queries)
                    # queries = random.sample(range(0, len(dataset_options)), query_length)
                    slide_index = 0
                    for slide_inside in mcd_file.slides:
                        if predefined_indices is None:
                            if num_queries > len(slide_inside.acquisitions):
                                query_selection = range(0, len(slide_inside.acquisitions))
                            else:
                                query_selection = random.sample(range(0, len(slide_inside.acquisitions)), num_queries)
                        if isinstance(query_selection, dict):
                            if 'indices' in query_selection:
                                num_queries = len(query_selection['indices'])
                                query_selection = [i for i in query_selection['indices'] if \
                                                   len(slide_inside.acquisitions) > i >= 0]
                            elif 'names' in query_selection:
                                acq_names = [acq.description for acq in slide_inside.acquisitions]
                                num_queries = len(query_selection['names'])
                                query_selection = [acq_names.index(name) for name in query_selection['names']]
                        for query in query_selection:
                            acq = slide_inside.acquisitions[query]
                            if f"{basename}+++slide{slide_index}+++{acq.description}" not in rois_exclude:
                                channel_names = acq.channel_names
                                channel_index = 0
                                img = mcd_file.read_acquisition(acq)
                                acq_image = []
                                for channel in img:
                                    # if the channel is in the current blend, use it
                                    if channel_names[channel_index] in currently_selected_channels and \
                                        channel_names[channel_index] in blend_dict.keys():
                                        with_preset = apply_preset_to_array(channel,
                                                        blend_dict[channel_names[channel_index]])
                                        recoloured = np.array(recolour_greyscale(with_preset,
                                                                    blend_dict[channel_names[channel_index]][
                                                                        'color'])).astype(np.float32)
                                        acq_image.append(recoloured)
                                    channel_index += 1
                                label = f"{basename}+++slide{slide_index}+++{acq.description}"
                                matched_mask = match_mask_name_with_roi(label, mask_dict, dataset_options)
                                summed_image = sum([image for image in acq_image]).astype(np.float32)
                                summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)
                                # find a matched mask and check if the dimensions are compatible. If so, add to the gallery
                                if matched_mask is not None and matched_mask in mask_dict.keys() and \
                                validate_mask_shape_matches_image(summed_image, mask_dict[matched_mask]["boundary"]):
                                    # TODO: establish the ability to subset the mask to just the cells from the query
                                    # for each ROI
                                    # requires reverse matching the sample or description to the ROI name in the app
                                    # if the query cell is list exists, subset the mask
                                    if query_cell_id_lists is not None:
                                        sam_names = list(query_cell_id_lists.keys())
                                        # match the sample name in te quant sheet to the matched mask name
                                        sam_name = match_mask_name_to_quantification_sheet_roi(matched_mask, sam_names)
                                        if sam_name is not None:
                                            mask_to_use = subset_mask_outline_using_cell_id_list(
                                            mask_dict[matched_mask]["boundary"], mask_dict[matched_mask]["raw"],
                                            query_cell_id_lists[sam_name])
                                        else:
                                            mask_to_use = mask_dict[matched_mask]["boundary"]
                                    else:
                                        mask_to_use = mask_dict[matched_mask]["boundary"]
                                    summed_image = cv2.addWeighted(summed_image.astype(np.uint8), 1,
                                            mask_to_use.astype(np.uint8), 1, 0).astype(np.uint8)
                                roi_images[label] = summed_image
                                queries_obtained += 1
                            else:
                                num_queries += 1
                            if len(roi_images) == num_queries:
                                break
                        else:
                            slide_index += 1
                            continue
                        break
                    else:
                        continue
                    break
            return roi_images
        else:
            return None
    # return roi_images
    except (KeyError, AssertionError):
        return None
