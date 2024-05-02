from pathlib import Path
from ccramic.utils.pixel_level_utils import (
    apply_preset_to_array,
    recolour_greyscale,
    apply_filter_to_array)
from ccramic.parsers.pixel_level_parsers import convert_rgb_to_greyscale
from ccramic.utils.cell_level_utils import validate_mask_shape_matches_image
from ccramic.utils.roi_utils import subset_mask_outline_using_cell_id_list
from ccramic.parsers.cell_level_parsers import (
    match_mask_name_with_roi,
    match_mask_name_to_quantification_sheet_roi)
from readimc import MCDFile, TXTFile
import random
import numpy as np
import cv2
from tifffile import TiffFile
from typing import Union


class RegionThumbnail:
    """
    Generates a preview image for a series of ROIs, selected either by name, or randomly based on the number
    of queries passed
    Thumbnail images will be appended to the ROI gallery with identical blend parameters to the current ROI
    """
    def __init__(self, session_config, blend_dict, currently_selected_channels, num_queries=5, rois_exclude=None,
                        predefined_indices=None, mask_dict=None, dataset_options=None, query_cell_id_lists=None,
                        global_apply_filter=False, global_filter_type="median", global_filter_val=3,
                        global_filter_sigma=1, delimiter: str="+++", use_greyscale: bool=False,
                        dimension_limit: Union[int, float, None]=None):
        self.session_config = session_config
        try:
            self.file_list = [file for file in self.session_config['uploads']]
        except KeyError:
            self.file_list = []
        self.blend_dict = blend_dict
        self.currently_selected_channels = currently_selected_channels
        self.num_queries = num_queries
        self.rois_exclude = rois_exclude if rois_exclude is not None else []
        self.predefined_indices = predefined_indices
        self.mask_dict = mask_dict
        self.dataset_options = dataset_options if dataset_options else []
        self.query_cell_id_lists = query_cell_id_lists if predefined_indices is not None else None
        self.global_filter_apply = global_apply_filter
        self.global_filter_type = global_filter_type
        self.global_filter_val = global_filter_val
        self.global_filter_sigma = global_filter_sigma
        self.delimiter = delimiter
        self.query_selection = None
        self.use_greyscale = use_greyscale
        # do not use the dimension limit if querying from the quantification
        self.dim_limit = dimension_limit if (dimension_limit and not self.predefined_indices) else 0

        if self.predefined_indices is not None:
            self.query_selection = predefined_indices
        self.roi_images = {}
        if predefined_indices is None:
            random.shuffle(self.file_list)
        if self.file_list is not None and len(self.file_list) > 0:
            self.queries_obtained = 0
            for file_path in self.file_list:
                if str(file_path).endswith('.mcd'):
                    self.additive_thumbnail_from_mcd(file_path)
                elif str(file_path).endswith('.tiff') or str(file_path).endswith('.tif'):
                    self.additive_thumbnail_from_tiff(file_path)
                elif str(file_path).endswith('.txt'):
                    self.additive_thumbnail_from_txt(file_path)
                if len(self.roi_images) >= self.num_queries:
                    break

    def additive_thumbnail_from_mcd(self, file_path):
        basename = str(Path(file_path).stem)
        with MCDFile(file_path) as mcd_file:
            # generate a random number of roi indices to query
            # query_length = min(len(dataset_options), num_queries)
            # queries = random.sample(range(0, len(dataset_options)), query_length)
            slide_index = 0
            for slide_inside in mcd_file.slides:
                if self.predefined_indices is None:
                    if self.num_queries > len(slide_inside.acquisitions):
                        self.query_selection = range(0, len(slide_inside.acquisitions))
                    else:
                        self.query_selection = random.sample(range(0,
                                                len(slide_inside.acquisitions)), self.num_queries)
                elif isinstance(self.query_selection, dict):
                    if 'indices' in self.query_selection:
                        self.num_queries = len(self.query_selection['indices'])
                        self.query_selection = [i for i in self.query_selection['indices'] if \
                                           len(slide_inside.acquisitions) > i >= 0]
                    elif 'names' in self.query_selection:
                        acq_names = [f"{str(acq.description)}_{str(acq.id)}" for acq in slide_inside.acquisitions]
                        self.num_queries = len(self.query_selection['names'])
                        try:
                            self.query_selection = [acq_names.index(name) for name in self.query_selection['names']]
                        except ValueError:
                            self.query_selection = []
                for query in self.query_selection:
                    try:
                        acq = slide_inside.acquisitions[query]
                        if f"{basename}{self.delimiter}slide{slide_index}{self.delimiter}" \
                           f"{str(acq.description)}_{str(acq.id)}" not in self.rois_exclude and \
                                (acq.height_px >= self.dim_limit and acq.width_px >= self.dim_limit):
                            channel_names = acq.channel_names
                            channel_index = 0
                            img = mcd_file.read_acquisition(acq)
                            acq_image = []
                            for channel in img:
                                # if the channel is in the current blend, use it
                                if channel_names[channel_index] in self.currently_selected_channels and \
                                        channel_names[channel_index] in self.blend_dict.keys():
                                    with_preset = apply_preset_to_array(channel,
                                                                        self.blend_dict[channel_names[channel_index]])
                                    colour_use = self.blend_dict[channel_names[channel_index]]['color'] if not \
                                        self.use_greyscale else '#FFFFFF'
                                    recoloured = np.array(recolour_greyscale(with_preset, colour_use)).astype(
                                        np.float32)
                                    acq_image.append(recoloured)
                                channel_index += 1
                            label = f"{basename}{self.delimiter}slide{slide_index}{self.delimiter}" \
                                    f"{str(acq.description)}_{str(acq.id)}"
                            self.process_additive_image(acq_image, label)
                        else:
                            additional_query = None
                            # use the look counter to make sure that once all of the indices are searched,
                            # it will just return what has been made nad not search indefinitely
                            look_counter = 0
                            while (additional_query is None or additional_query in self.query_selection) and \
                                    look_counter < len(slide_inside.acquisitions):
                                additional_query = random.sample(range(0,
                                                    len(slide_inside.acquisitions)), 1)
                                additional_query = additional_query[0] if isinstance(additional_query, list) else \
                                    additional_query
                                look_counter += 1
                                if additional_query not in self.query_selection:
                                    self.query_selection.append(additional_query)
                                    self.num_queries += 1
                                    break
                    except OSError:
                        pass
                    if len(self.roi_images) >= self.num_queries:
                        break
                else:
                    slide_index += 1
                    continue
            # else:
            #     continue
            # break

    def additive_thumbnail_from_tiff(self, tiff_filepath):
        # set the channel label by parsing through the dataset options to find a partial match of filename
        basename = str(Path(tiff_filepath).stem)
        label = self.parse_thumbnail_label_from_filepath(basename)
        matched_mask = match_mask_name_with_roi(label, self.mask_dict, self.dataset_options, self.delimiter)
        # if queried from the UMAP plot, restrict to only those with a match in the query selection
        if self.query_selection and 'names' in self.query_selection:
            query_list = self.query_selection['names']
            label = label if match_mask_name_to_quantification_sheet_roi(matched_mask, query_list) else None
        if label and label not in self.rois_exclude:
            with TiffFile(tiff_filepath) as tif:
                # add conditional to check if the tiff dimensions meet the threshold
                if tif.pages[0].shape[0] >= self.dim_limit and tif.pages[0].shape[1] >= self.dim_limit:
                    acq_image = []
                    channel_index = 1
                    for page in tif.pages:
                        channel_name = str(f"channel_{channel_index}")
                        if channel_name in self.currently_selected_channels and \
                                channel_name in self.blend_dict.keys():
                            with_preset = apply_preset_to_array(convert_rgb_to_greyscale(page.asarray()),
                                                                self.blend_dict[channel_name])
                            colour_use = self.blend_dict[channel_name]['color'] if not \
                                self.use_greyscale else '#FFFFFF'
                            recoloured = np.array(recolour_greyscale(with_preset, colour_use)).astype(np.float32)
                            acq_image.append(recoloured)
                        channel_index += 1
                    self.process_additive_image(acq_image, label)

    def additive_thumbnail_from_txt(self, txt_filepath):
        basename = str(Path(txt_filepath).stem)
        label = self.parse_thumbnail_label_from_filepath(basename)
        if label not in self.rois_exclude:
            with TXTFile(txt_filepath) as acq_text_read:
                image_index = 1
                txt_channel_names = acq_text_read.channel_names
                acq_image = []
                acq_read = acq_text_read.read_acquisition()
                # IMP: txt files contain the channel number as the first dimension, not the last
                acq_dims = (acq_read.shape[1], acq_read.shape[2]) if len(acq_read.shape) >= 3 else \
                    (acq_read.shape[0], acq_read.shape[1])
                if acq_dims[0] >= self.dim_limit and acq_dims[1] >= self.dim_limit:
                    for image in acq_read:
                        channel_name = txt_channel_names[image_index - 1]
                        if channel_name in self.currently_selected_channels and \
                                channel_name in self.blend_dict.keys():
                            with_preset = apply_preset_to_array(image,
                                                                self.blend_dict[channel_name])
                            colour_use = self.blend_dict[channel_name]['color'] if not \
                                self.use_greyscale else '#FFFFFF'
                            recoloured = np.array(recolour_greyscale(with_preset, colour_use)).astype(np.float32)
                            acq_image.append(recoloured)
                        image_index += 1
                    self.process_additive_image(acq_image, label)

    def parse_thumbnail_label_from_filepath(self, file_basename):
        """
        Parse a list of imported dataset options and parse the list based on a file basename
        """
        label = "None"
        for dataset in self.dataset_options:
            if file_basename in dataset:
                label = dataset
        return label

    def process_additive_image(self, image_list: list, label: str):
        """
        Process the additive image prior to appending to the gallery, includes matching the mask and
        applying global filters
        """
        matched_mask = match_mask_name_with_roi(label, self.mask_dict, self.dataset_options, self.delimiter)
        summed_image = sum([image for image in image_list]).astype(np.float32)
        summed_image = apply_filter_to_array(summed_image, self.global_filter_apply,
                                             self.global_filter_type, self.global_filter_val,
                                             self.global_filter_sigma)
        summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)
        # find a matched mask and check if the dimensions are compatible. If so, add to the gallery
        if matched_mask is not None and matched_mask in self.mask_dict.keys() and \
                validate_mask_shape_matches_image(summed_image, self.mask_dict[matched_mask]["boundary"]):
            # requires reverse matching the sample or description to the ROI name in the app
            # if the query cell is list exists, subset the mask
            if self.query_cell_id_lists is not None:
                sam_names = list(self.query_cell_id_lists.keys())
                # match the sample name in te quant sheet to the matched mask name
                # TODO: update logic here when the names do not match exactly from quantification to mask
                # in-app quantification
                sam_name = match_mask_name_to_quantification_sheet_roi(matched_mask, sam_names)
                if sam_name is not None:
                    mask_to_use = subset_mask_outline_using_cell_id_list(
                        self.mask_dict[matched_mask]["raw"], self.mask_dict[matched_mask]["raw"],
                        self.query_cell_id_lists[sam_name])
                else:
                    mask_to_use = self.mask_dict[matched_mask]["boundary"]
            else:
                mask_to_use = self.mask_dict[matched_mask]["boundary"]
            mask_to_use = np.where(mask_to_use > 0, 255, 0)
            summed_image = cv2.addWeighted(summed_image.astype(np.uint8), 1,
                                           mask_to_use.astype(np.uint8), 1, 0).astype(np.uint8)
        self.roi_images[label] = summed_image
        self.queries_obtained += 1

    def get_image_dict(self):
        return self.roi_images if len(self.roi_images) > 0 else None
