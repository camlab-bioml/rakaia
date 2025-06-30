"""Module containing functions and classes to generate region-level thumbnails
for gallery rendering
"""
from typing import Union
from functools import partial
import os
from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image
from readimc.data import Slide, Acquisition
from tifffile import TiffFile
from readimc import MCDFile, TXTFile
import numexpr as ne
import anndata as ad
from rakaia.utils.pixel import (
    apply_preset_to_array,
    recolour_greyscale,
    apply_filter_to_array, set_array_storage_type_from_config)
from rakaia.utils.object import validate_mask_shape_matches_image, convert_mask_to_object_boundary
from rakaia.utils.roi import subset_mask_outline_using_cell_id_list
from rakaia.parsers.object import (
    ROIMaskMatch,
    match_mask_name_to_quantification_sheet_roi)
from rakaia.parsers.spatial import (
    spatial_canvas_dimensions, spatial_grid_single_marker,
    get_spatial_spot_radius,
    is_spot_based_spatial,
    is_spatial_dataset)

class RegionThumbnail:
    """
    Generates a preview image for a series of ROIs, selected either by name, or randomly based on the number
    of queries passed. Thumbnail images will be appended to the ROI gallery with identical
    blend parameters to the current ROI.

    :param session_config: Dictionary of session parameters parsed from CLI
    :param blend_dict: Dictionary of channel blend parameters
    :param currently_selected_channels: List of channels in the current canvas blend
    :param num_queries: Number of ROI thumbnails to return from random search. Default is 5
    :param rois_exclude: List of ROI identifiers to exclude
    :param predefined_indices: list of indices for ROIS to include, if querying from quantification results
    :param mask_dict: Dictionary of imported mask arrays
    :param dataset_options: List of ROIs that can be loaded in the current session
    :param query_cell_id_lists: Dictionary of lists of mask object IDs to subset ROI masks
    :param global_apply_filter: Whether a global filter has been applied
    :param global_filter_type: String specifying a global gaussian or median blur
    :param global_filter_val: Kernel size for the global gaussian or median blur
    :param global_filter_sigma: If global gaussian blur is applied, set the sigma value
    :param delimiter: String to split ROI identifiers on. Default is `+++`
    :param use_greyscale: Whether grey scaling should be used for image thumbnails. Default is False
    :param dimension_min: Minimum dimension in pixels for an ROI thumbnail to be generated.
    :param dimension_max: Maximum dimension in pixels for an ROI thumbnail to be generated.
    :param roi_keyword: String keyword used to search for ROI names.
    :param single_channel_view: Whether the thumbnail should be used to preview a single greyscale channel thumbnail.
    :param spatial_radius: Optional value for user-driven spatial marker radius. By default, inferred from the dataset.
    :param enable_masks: Whether to use a matching mask, if found, for each thumbnail. Default: True
    :param use_scaling: Whether to use the percentile scaling from the blend dictionary or not (Default: True)
    :return: None
    """
    # define string attribute matches for the partial
    MATCHES = {".mcd": "mcd", ".tiff": "tiff", ".tif": "tiff", ".txt": "txt", ".h5ad": "h5ad"}

    def __init__(self, session_config, blend_dict, currently_selected_channels, num_queries=5, rois_exclude=None,
                 predefined_indices=None, mask_dict=None, dataset_options=None, query_cell_id_lists=None,
                 global_apply_filter=False, global_filter_type="median", global_filter_val=3,
                 global_filter_sigma=1, delimiter: str="+++", use_greyscale: bool=False,
                 dimension_min: Union[int, float, None]=None,
                 dimension_max: Union[int, float, None]=None,
                 roi_keyword: str=None,
                 single_channel_view: bool=False, spatial_radius: Union[int, None]=None,
                 enable_masks: bool=True,
                 use_scaling: bool=True, arr_type: str="float"):

        self.file_list = None
        self.mcd = partial(self.additive_thumbnail_from_mcd)
        self.tiff = partial(self.additive_thumbnail_from_tiff)
        self.txt = partial(self.additive_thumbnail_from_txt)
        self.h5ad = partial(self.additive_thumbnail_from_h5ad)
        self.session_config = session_config

        self.set_imported_files()
        self.blend_dict = blend_dict
        self.currently_selected_channels = currently_selected_channels
        self.num_queries = num_queries if not predefined_indices else 1e6
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
        self.single_channel_view = single_channel_view
        # do not use the dimension limit if querying from the quantification
        self.dim_min = self.set_dimension_min(dimension_min)
        self.dim_max = self.set_dimension_max(dimension_max)
        self.keyword = self.set_query_keywords(roi_keyword)
        self.spot_size = spatial_radius
        # if random query, allow mask toggle, otherwise always include the mask
        self.enable_masks = enable_masks if not predefined_indices else True
        self.use_scaling = use_scaling
        self.arr_type = arr_type if arr_type else "float"

        # if querying using objects, track the order (descending) across multi-ROI files
        self.order = {}

        self.set_keyword_with_defined_indices()
        self.set_selection_using_defined_indices(predefined_indices)

        self.roi_images = {}
        self.shuffle_files_without_defined_indices(predefined_indices)

        if self.file_list is not None and len(self.file_list) > 0:
            self.queries_obtained = 0
            for file_path in self.file_list:
                if len(self.roi_images) >= self.num_queries:
                    break
                filename, file_extension = os.path.splitext(file_path)
                # call the additive thumbnail partial function with the corresponding extension
                try:
                    getattr(self, self.MATCHES[file_extension])(file_path)
                except (KeyError, ValueError): pass

    def set_imported_files(self):
        """
        Set the files for parsing based on the session config dictionary uploads. If a key error occurs, pass

        :return: None
        """
        try:
            self.file_list = [file for file in self.session_config['uploads']]
        except KeyError:
            self.file_list = []

    @staticmethod
    def set_query_keywords(roi_keyword: str=None):
        """
        Set the list of query keywords by comma separating an expression passed by the user

        :param roi_keyword: String combination of comma separated keywords to query
        :return: list of query keywords by comma separating an expression passed by the user
        """
        return [word.strip() for word in roi_keyword.split(",")] if (roi_keyword and "," in roi_keyword) else \
            ([roi_keyword] if roi_keyword else None)

    def set_dimension_min(self, dimension_min: Union[int, float]=None):
        """
        Set the query dimension minimum based on the value passed and the type of query being done.

        :param dimension_min: The numerical query value for the ROI size minimum (both dimensions).
        :return: Dimension min that is either the query value, or 0 if predefined ROIs are used.
        """
        return dimension_min if (dimension_min and not self.predefined_indices) else 0

    def set_dimension_max(self, dimension_max: Union[int, float]=None):
        """
        Set the query dimension minimum based on the value passed and the type of query being done.

        :param dimension_max: The numerical query value for the ROI size maximum (both dimensions).
        :return: Dimension max that is either the query value, or 1e6 if predefined ROIs are used.
        """
        return dimension_max if (dimension_max and not self.predefined_indices) else 1e6

    def shuffle_files_without_defined_indices(self, predefined_indices):
        """
        If predefined indices are not used, shuffle the filenames for a random generation

        :return: None
        """
        if predefined_indices is None:
            random.shuffle(self.file_list)

    def set_keyword_with_defined_indices(self):
        """
        Do not use a query keyword if querying from predefined indices

        :return: None
        """
        if self.predefined_indices:
            self.keyword = None

    def set_selection_using_defined_indices(self, predefined_indices):
        """
        Set the query selection to the predefined indices, if they exist

        :return: None
        """
        if self.predefined_indices is not None:
            self.query_selection = predefined_indices

    def set_mcd_query_selection(self, mcd_slide, queries_by_name: list=None,
                                pad_indices: bool=True):
        """
        Set the mcd query selection for one or multiple mcd files, checking the query number against the dataset size.

        :param mcd_slide: Slide object from readimc from the currently parsed mcd file
        :param queries_by_name: An existing list of mcd slide queries. Useful for when multiple mcd files are queried
        :param pad_indices: Whether to offset the indices by 1 to match the way mcd ROIs are indexed. Default is True
        :return: None
        """
        if self.predefined_indices is None:
            roi_indices = [int(acq.id) for acq in mcd_slide.acquisitions]
            if self.num_queries > len(mcd_slide.acquisitions):
                # self.query_selection = range(1, (len(mcd_slide.acquisitions) + 1))
                self.query_selection = roi_indices
            else:
                self.query_selection = random.sample(roi_indices, self.num_queries)
        elif isinstance(self.query_selection, dict):
            if 'indices' in self.query_selection:
                pad = 1 if pad_indices else 0
                self.num_queries = len(self.query_selection['indices'])
                self.query_selection = [int(i + pad) for i in self.query_selection['indices'] if \
                                        len(mcd_slide.acquisitions) > i >= 0]
            elif 'names' in self.query_selection:
                self.num_queries = len(self.query_selection['names'])
                queries_by_name = [] if queries_by_name is None else queries_by_name
                for name in self.query_selection['names']:
                    queries_by_name.append(str(name))
                self.query_selection = queries_by_name

    def roi_within_dimension_threshold(self, roi_height, roi_width) -> bool:
        """
        :return: If the ROI height and width dimensions fit within the provided min and max query dimensions
        """
        return (roi_height >= self.dim_min and roi_width >= self.dim_min) and \
                (roi_height <= self.dim_max and roi_width <= self.dim_max)

    @staticmethod
    def set_mcd_acquisition_from_query(slide_inside: Slide, query: Union[int, str]) -> Union[Acquisition, None]:
        """
        Set the slide acquisition from the query from either an integer based index, or an acquisition name.
        Using acquisition string descriptions for querying permits retrieval of ROIs across multiple MCd files
        from a common quantification dataset.

        :param slide_inside: Current MCD slide
        :param query: Integer index for the acquisition, or a query string name to match the acquisition description
        :return: Slide acquisition to read for thumbnail generation.
        """
        # need to get the list of roi ids in the slide, as the integers may be non-consecutive
        rois_in_slide = [acq.id for acq in slide_inside.acquisitions]
        if isinstance(query, int):
            try:
                acq = slide_inside.acquisitions[rois_in_slide.index(int(query))]
            except (IndexError, KeyError): acq = None
        else:
            try:
                acq_list = [f"{acq.description}_{acq.id}" for acq in slide_inside.acquisitions]
                acq = slide_inside.acquisitions[acq_list.index(query)]
            except (KeyError, TypeError, IndexError, ValueError):
                return None
        return acq


    def set_mcd_acquisition_by_index(self, slide_inside: Slide,
                                     acq: Union[Acquisition, None], query: str,
                                     mcd_basename: str):
        """
        Set the acquisition by index if the acquisition has not been successfully set by the query string

        :param slide_inside: Current MCD slide
        :param acq: Current acquisition selected, either an `Acquisition` class or `None`
        :param query: Integer index for the acquisition, or a query string name to match the acquisition description
        :param mcd_basename: String basename of the current mcd to which `slide_inside` belongs
        :return: Slide acquisition to read for thumbnail generation.
        """
        if not acq:
            # Only split at the last underscore instance
            filename, roi_index = query.rsplit('_', 1)
            if filename == mcd_basename:
                acq = self.set_mcd_acquisition_from_query(slide_inside, int(roi_index))
        return acq

    def apply_preset_based_on_view(self, arr: np.array, channel_name: str):
        """
        Apply a single channel preset if scaling is to be used, and single channel view is not enabled
        (Single channel scaling processed separately).

        :param arr: Numpy channel array
        :param channel_name: Channel identifier in the session dictionary

        :return: Scaled numpy channel array based on condition
        """
        return apply_preset_to_array(arr.astype(set_array_storage_type_from_config(self.arr_type)),
                                     self.blend_dict[channel_name]) if \
            self.use_scaling else arr.astype(set_array_storage_type_from_config(self.arr_type))

    def set_colour(self, channel_name: str):
        """
        Set the channel recolour to use

        :param channel_name: Internal channel identifier

        :return: RGB colour value to apply for recolouring
        """
        return self.blend_dict[channel_name]['color'] if not \
            self.use_greyscale else '#FFFFFF'

    def apply_colouring_based_on_view(self, arr: np.array, colour: str):
        """
        Apply a single channel recolouring if single channel view is not used
        (single channel view is greyscale)

        :param arr: Numpy channel array
        :param colour: RGB colour to use for colouring

        :return: Recoloured channel array
        """
        return np.array(recolour_greyscale(arr, colour)).astype(
            np.float32) if not self.single_channel_view else arr.astype(
            np.float32)

    def single_channel_process(self, arr: np.array, channel_name: str):
        """
        Define the steps for single channel processing. Includes scaling and recolouring

        :param arr: Numpy channel array
        :param channel_name: Channel identifier in the session dictionary
        """
        with_preset = self.apply_preset_based_on_view(arr, channel_name)
        colour_use = self.set_colour(channel_name)
        return self.apply_colouring_based_on_view(with_preset, colour_use)

    def additive_thumbnail_from_mcd(self, file_path):
        """
        Generate one or more image thumbnails from a mcd file.

        :param file_path: Path to a mcd file.
        :return: None
        """
        basename = str(Path(file_path).stem)
        queries_by_name = []
        with MCDFile(file_path) as mcd_file:
            slide_index = 0
            for slide_inside in mcd_file.slides:
                self.set_mcd_query_selection(slide_inside, queries_by_name)
                for query in self.query_selection:
                    try:
                        acq = self.set_mcd_acquisition_from_query(slide_inside, query)
                        acq = self.set_mcd_acquisition_by_index(slide_inside, acq, query, basename)
                        roi_identifier = f"{basename}{self.delimiter}slide{slide_index}{self.delimiter}" \
                           f"{str(acq.description)}_{str(acq.id)}"
                        if self.roi_keyword_in_roi_identifier(roi_identifier) and roi_identifier \
                                not in self.rois_exclude and \
                                self.roi_within_dimension_threshold(acq.height_px, acq.width_px):
                            channel_names = acq.channel_names
                            channel_index = 0
                            img = mcd_file.read_acquisition(acq, strict=False)
                            acq_image = []
                            for channel in img:
                                # if the channel is in the current blend, use it
                                if channel_names[channel_index] in self.currently_selected_channels and \
                                        channel_names[channel_index] in self.blend_dict.keys():
                                    recoloured = self.single_channel_process(channel, channel_names[channel_index])
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
                                roi_indices = [acq.id for acq in slide_inside.acquisitions if acq.id not in self.query_selection]
                                additional_query = random.sample(roi_indices, 1)
                                additional_query = additional_query[0] if isinstance(additional_query, list) else \
                                    additional_query
                                look_counter += 1
                                if additional_query not in self.query_selection:
                                    self.query_selection.append(additional_query)
                                    # self.num_queries += 1
                                    break
                    except (OSError, ValueError, IndexError, KeyError, AttributeError):
                        pass
                    if len(self.roi_images) >= self.num_queries:
                        break
                else:
                    slide_index += 1
                    continue
            # else:
            #     continue
            # break

    def label_from_named_query(self, filepath: str):
        """
        Set the ROI label based on its inclusion in the named query list. If the filepath is determined
        to be part of the named query, generate the label. Otherwise, set the label to `None`

        :param filepath: String for the current filepath to evaluate.
        """
        basename = str(Path(filepath).stem)
        label = self.parse_thumbnail_label_from_filepath(basename)
        matched_mask = ROIMaskMatch(label, self.mask_dict, self.dataset_options, self.delimiter).get_match()
        # if queried from the UMAP plot, restrict to only those with a match in the query selection
        if self.query_selection and 'names' in self.query_selection:
            query_list = self.query_selection['names']
            label = label if match_mask_name_to_quantification_sheet_roi(matched_mask, query_list) else None
        return label

    def additive_thumbnail_from_tiff(self, tiff_filepath):
        """
        Generate an image thumbnail from a tiff file.

        :param tiff_filepath: Path to a tiff file.
        :return: None
        """
        label = self.label_from_named_query(tiff_filepath)
        if label and label not in self.rois_exclude and (self.roi_keyword_in_roi_identifier(label) or
                self.roi_keyword_in_roi_identifier(tiff_filepath)):
            with TiffFile(tiff_filepath) as tif:
                # add conditional to check if the tiff dimensions meet the threshold
                if self.roi_within_dimension_threshold(tif.pages[0].shape[0], tif.pages[0].shape[1]):
                    acq_image = []
                    channel_index = 1
                    for page in tif.pages:
                        channel_name = str(f"channel_{channel_index}")
                        if channel_name in self.currently_selected_channels and \
                                channel_name in self.blend_dict.keys():
                            recoloured = self.single_channel_process(page.asarray(), channel_name)
                            acq_image.append(recoloured)
                        channel_index += 1
                    self.process_additive_image(acq_image, label)

    def additive_thumbnail_from_txt(self, txt_filepath):
        """
        Generate an image thumbnail from a txt file.

        :param txt_filepath: Path to a txt file.
        :return: None
        """
        label = self.label_from_named_query(txt_filepath)
        if label not in self.rois_exclude and (self.roi_keyword_in_roi_identifier(label) or
                self.roi_keyword_in_roi_identifier(txt_filepath)):
            with TXTFile(txt_filepath) as acq_text_read:
                image_index = 1
                txt_channel_names = acq_text_read.channel_names
                acq_image = []
                acq_read = acq_text_read.read_acquisition()
                # IMP: txt files contain the channel number as the first dimension, not the last
                acq_dims = (acq_read.shape[1], acq_read.shape[2]) if len(acq_read.shape) >= 3 else \
                    (acq_read.shape[0], acq_read.shape[1])
                if self.roi_within_dimension_threshold(acq_dims[0], acq_dims[1]):
                    for image in acq_read:
                        channel_name = txt_channel_names[image_index - 1]
                        if channel_name in self.currently_selected_channels and \
                                channel_name in self.blend_dict.keys():
                            recoloured = self.single_channel_process(image, channel_name)
                            acq_image.append(recoloured)
                        image_index += 1
                    self.process_additive_image(acq_image, label)

    def additive_thumbnail_from_h5ad(self, h5ad_filepath):
        """
        Generate an image thumbnail from an .h5ad Anndata file. Current technologies supported are:
        10X Visium (V1, V2, HD), Xenium

        :param h5ad_filepath: Path to an .h5ad Anndata file.
        :return: None
        """
        basename = str(Path(h5ad_filepath).stem)
        label = self.parse_thumbnail_label_from_filepath(basename)
        if label not in self.rois_exclude and (self.roi_keyword_in_roi_identifier(label) or
                                               self.roi_keyword_in_roi_identifier(h5ad_filepath)):
            adata = ad.read_h5ad(h5ad_filepath)
            # get the visium anndata dimensions
            acq_image = []
            grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(adata)
            if (self.roi_within_dimension_threshold(grid_width, grid_height) and
                    (is_spot_based_spatial(adata) or is_spatial_dataset(adata))):
                spot_size = get_spatial_spot_radius(adata, self.spot_size)
                for marker in self.currently_selected_channels:
                    if marker in self.blend_dict.keys():
                        recoloured = self.single_channel_process(spatial_grid_single_marker(
                            adata, marker, spot_size), marker)
                        acq_image.append(recoloured)
                self.process_additive_image(acq_image, label)


    def parse_thumbnail_label_from_filepath(self, file_basename) -> str:
        """
        Parse a list of imported dataset options and parse the list based on a file basename

        :param file_basename: The file basename for the current filepath.
        :return: String label linking the file basename to the internal ROI identifier, or `None` if no match is found.
        """
        label = None
        for dataset in self.dataset_options:
            if file_basename in dataset:
                label = dataset
        return label

    @staticmethod
    def summed_image_from_list(image_list: list) -> Union[np.array, np.ndarray]:
        """
        Generate a summed channel image for each thumbnail from a list of channel arrays (un-clipped)

        :param image_list: List of channel arrays to be blended
        :return: Summed image array with a `numpy` float32 dtype
        """
        if image_list and len(image_list) >= 1:
            image_shape = image_list[0].shape
            image = np.zeros(image_shape)
            for elem in image_list:
                image = ne.evaluate("image + elem")
            return image.astype(np.float32)
        return None

    def process_additive_image(self, image_list: list, label: str):
        """
        Process the additive image prior to appending to the gallery, includes matching the mask and
        applying global filters

        :param image_list: List of numpy channel arrays to blend
        :param label: Image label to be displayed above the thumbnail.
        :return: None
        """
        if len(self.roi_images) < self.num_queries and image_list:
            summed_image = self.summed_image_from_list(image_list)
            if not self.single_channel_view:
                matched_mask = ROIMaskMatch(label, self.mask_dict, self.dataset_options, self.delimiter).get_match()
                summed_image = apply_filter_to_array(summed_image, self.global_filter_apply,
                                                     self.global_filter_type, self.global_filter_val,
                                                     self.global_filter_sigma)
                summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)
                # find a matched mask and check if the dimensions are compatible. If so, add to the gallery
                if matched_mask is not None and matched_mask in self.mask_dict.keys() and \
                        validate_mask_shape_matches_image(summed_image, self.mask_dict[matched_mask]["raw"]) and \
                    self.enable_masks:
                    # requires reverse matching the sample or description to the ROI name in the app
                    # if the query cell is list exists, subset the mask
                    if self.query_cell_id_lists is not None:
                        sam_names = list(self.query_cell_id_lists.keys())
                        # match the sample name in te quant sheet to the matched mask name
                        # logic here should be flexible with different mask and sample names in the quantification sheet
                        sam_name = match_mask_name_to_quantification_sheet_roi(matched_mask, sam_names)
                        if sam_name is not None:
                            mask_to_use = subset_mask_outline_using_cell_id_list(
                                self.mask_dict[matched_mask]["raw"], self.mask_dict[matched_mask]["raw"],
                                self.query_cell_id_lists[sam_name])
                            # keep track of how many objects in the ROI
                            self.order[label] = len(self.query_cell_id_lists[sam_name])
                        else:
                            mask_to_use = np.array(Image.fromarray(
                                convert_mask_to_object_boundary(
                                    self.mask_dict[matched_mask]["raw"])).convert('RGB'))
                    else:
                        mask_to_use = np.array(Image.fromarray(
                                convert_mask_to_object_boundary(
                                    self.mask_dict[matched_mask]["raw"])).convert('RGB'))
                    mask_to_use = np.where(mask_to_use > 0, 255, 0)
                    summed_image = cv2.addWeighted(summed_image.astype(np.uint8), 1,
                                                   mask_to_use.astype(np.uint8), 1, 0).astype(np.uint8)
            self.roi_images[label] = summed_image
            self.queries_obtained += 1

    def get_image_dict(self) -> dict:
        """
        Get the dictionary of ROI thumbnail images

        :return: Dictionary where keys are ROI identifiers, and values are the blended images for each thumbnail.
        """
        if self.order:
            # if querying using objects, reorder them based on descending number across files
            order_rois = sorted(self.order, key=self.order.get, reverse=True)
            image_order = {key: self.roi_images[key] for key in order_rois}
            return image_order if len(image_order) > 0 else None
        return self.roi_images if len(self.roi_images) > 0 else None

    def roi_keyword_in_roi_identifier(self, roi_identifier: str=None):
        """
        Identify if an ROI keyword is in an ROI name.

        :param roi_identifier: String representation of the ROI inside the session.
        :Return: bool indicating keyword presence of not.
        """
        if roi_identifier and self.keyword:
            return any(elem in roi_identifier for elem in self.keyword)
        return True
