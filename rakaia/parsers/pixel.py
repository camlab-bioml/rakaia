"""Module containing functions and classes for parsing and validating raw images data
file imports and defining internal raw data hash maps
"""

from pathlib import Path
from typing import Union
from functools import partial
import os

import dash
import numpy as np
from tifffile import TiffFile
from readimc import MCDFile, TXTFile
from scipy.sparse import issparse, csc_matrix
import pandas as pd
from PIL import Image
import h5py
import anndata as ad

from rakaia.parsers.spatial import (
    spatial_canvas_dimensions,
    is_spot_based_spatial, is_spatial_dataset)
from rakaia.utils.pixel import (
    split_string_at_pattern,
    set_array_storage_type_from_config,
    get_default_channel_upper_bound_by_percentile)
from rakaia.utils.alert import PanelMismatchError

class NoAcquisitionsParsedError(Exception):
    """
    Passed when the session upload dictionary produces no viable ROIs
    """

def roi_requires_single_marker_load(pixel_counter: Union[np.array, np.ndarray, int],
                                    panel_length: int,
                                    lower_pixel_threshold: int=20000000, panel_size_threshold: int=50,
                                    upper_pixel_threshold: int=100000000):
    """
    Determines if an ROI is sufficiently large to require single marker loading. For example, if the channel array
    provided has more total pixels than `pixel_threshold` and is part of a panel size that is
    greater than `panel_size_threshold` (these ROIs typically will not fit in most memory).
    or if a single ROI dimension has more pixels than a 10000x10000 image
    """
    pixel_counter = int(pixel_counter.shape[0] * pixel_counter.shape[1]) if (
    isinstance(pixel_counter, np.ndarray)) else int(pixel_counter)
    return (pixel_counter >= lower_pixel_threshold and panel_length >= panel_size_threshold) or (
        pixel_counter >= upper_pixel_threshold)

class FileParser:
    """
    Parses a list of filepaths into a dictionary of image arrays, grouped by region (ROI) identifiers
    When using lazy loading, the dictionary will be created as a placeholder and a dataframe of ROI information
    will be created, but the dictionary will contain None values in place of image arrays. When lazy loading is
    turned off, greyscale image arrays are read into the dictionary slots with the numpy array type specified
    in `array_store_type`.

    :param filepaths: List of filepaths successfully imported into the session
    :param array_store_type: Specify the numpy dtype for channel arrays, 32-byte `float` or 16-byte `int`.
    :param lazy_load: Whether arrays should be loaded into memory only when their corresponding ROI is requested.
    :param single_roi_parse: When parsing mcd files, if only a single ROI should be parsed when using lazy loading.
    :param roi_name: When parsing mcd files and not using lazy loading, pass a single ROI name to pull from an mcd.
    :param internal_name: When not using lazy loading, retain the current ROI selection string
    :return: None
    """
    MATCHES = {".mcd": "mcd", ".tiff": "tiff", ".tif": "tiff", ".txt": "txt", ".h5": "h5",
               ".h5ad": "h5ad"}

    def __init__(self, filepaths: list, array_store_type: str="float", lazy_load: bool=True,
                 single_roi_parse: bool=True, roi_name: Union[str, None]=None,
                 internal_name: Union[str, None]=None, delimiter: str="+++"):

        self.check_for_valid_array_type(array_store_type)
        self.filepaths = [str(x) for x in filepaths]
        self.array_store_type = array_store_type
        self.image_dict = {}
        self.unique_image_names = []
        self.dataset_information_frame = {"ROI": [], "Dimensions": [], "Panel": []}
        self.lazy_load = lazy_load
        self.delimiter = delimiter
        self.panel_length = None
        if len(self.filepaths) > 0:
            self.image_dict['metadata'] = {}
            self.metadata_channels = []
            self.metadata_labels = []
            self.experiment_index = 0
            self.slide_index = 0
            self.acq_index = 0
            self.blend_config = None
            self.roi_name = roi_name
            self.internal_name = internal_name
            self.mcd = partial(self.parse_mcd)
            self.tiff = partial(self.parse_tiff, internal_name=self.internal_name)
            self.txt = partial(self.parse_txt, internal_name=self.internal_name)
            self.h5 = partial(self.parse_h5)
            self.h5ad = partial(self.parse_h5ad)

            for upload in self.filepaths:
                try:
                    # IMP: split reading a single mcd ROI from the entire mcd, as mcds can contain multiple ROIs
                    # this is currently unique to mcds: all other files have one ROI per file
                    filename, file_extension = os.path.splitext(upload)
                    self.check_for_valid_file_extension(file_extension, upload)
                    if upload.endswith('.mcd') and not lazy_load and single_roi_parse and \
                            None not in (roi_name, internal_name):
                        self.read_single_roi_from_mcd(upload, self.internal_name, self.roi_name)
                    else:
                        # call the additive thumbnail partial function with the corresponding extension
                        getattr(self, self.MATCHES[file_extension])(upload)
                except OSError:
                    pass

    @staticmethod
    def check_for_valid_array_type(array_store_type: str):
        """
        Check if the `array_store_type` passed is either a float or int string from the CLI options

        :param array_store_type: string of either `float` of `int`, usually provided through CLI on app initialization.
        """
        if array_store_type not in ["float", "int"]:
            raise TypeError("The array stored type must be one of float or int")

    def check_for_valid_file_extension(self, file_extension: str, upload: str):
        """
        Check if the provided file has an appropriate extension

        :param file_extension: Filetype extension of the uploaded file
        :param upload: filepath for the uploaded file
        """
        if file_extension not in list(self.MATCHES.keys()):
            raise TypeError(f"{upload} is not one of the supported image filetypes:\n"
                            ".mcd, .tiff, .txt, .h5, or .h5ad")

    def append_channel_identifier_to_collection(self, channel_name: str):
        """
        Append a unique channel identifier to the parsing collection if it doesn't yet exist.

        :param channel_name: string identifier for channel name
        :return: None
        """
        if channel_name not in self.unique_image_names:
            self.unique_image_names.append(channel_name)

    def append_channel_identifier_to_channel_list(self, channel_identifier: str):
        """
        Append a unique channel identifier to the metadata channel list if it doesn't yet exist.

        :param channel_identifier: string identifier for channel name
        :return: None
        """
        if channel_identifier not in self.metadata_channels:
            self.metadata_channels.append(channel_identifier)

    def append_channel_alias_to_label_list(self, channel_identifier: str):
        """
        Append a unique channel alias (internal session label) to the channel label list if it doesn't yet exist.
        Ensure that the number of labels does not exceed the number of unique keys

        :param channel_identifier: string identifier for channel name
        :return: None
        """
        if channel_identifier not in self.metadata_labels and \
                (len(self.metadata_labels) + 1) <= len(self.unique_image_names):
            self.metadata_labels.append(channel_identifier)

    def parse_h5(self, h5py_file):
        """
        Parse an .h5py ROI file generated from a previous session.

        :param h5py_file: path to a compatible .h5py file.
        :return: None
        """
        data_h5 = h5py.File(h5py_file, "r")
        self.blend_config = {}
        for roi in list(data_h5.keys()):
            self.image_dict[roi] = {}
            if roi not in ['metadata', 'metadata_columns']:
                channel_index = 1
                for channel in data_h5[roi]:
                    try:
                        # IMP: do not use lazy loading with h5 files as the filename is likely
                        # to be different from the internal experiment name due to renaming on export
                        self.image_dict[roi][channel] = data_h5[roi][channel]['image'][()].astype(
                        set_array_storage_type_from_config(self.array_store_type))
                        if channel_index == 1:
                            self.dataset_information_frame["ROI"].append(str(roi))
                            self.dataset_information_frame["Dimensions"].append(
                                f"{self.image_dict[roi][channel].shape[1]}x"
                                f"{self.image_dict[roi][channel].shape[0]}")
                            self.dataset_information_frame["Panel"].append(f"{len(data_h5[roi].keys())} markers")
                    except KeyError: pass
                    self.append_channel_identifier_to_collection(channel)
                    self.append_channel_identifier_to_channel_list(channel)
                    self.append_channel_alias_to_label_list(channel)
                    self.blend_config[channel] = {}
                    channel_index += 1
                    self.parse_h5_channel_blend_params(data_h5[roi], channel)
        meta_back = pd.DataFrame(data_h5['metadata'])
        for col in meta_back.columns:
            meta_back[col] = meta_back[col].str.decode("utf-8")
        try:
            meta_back.columns = [i.decode("utf-8") for i in data_h5['metadata_columns']]
        except KeyError: pass
        self.image_dict['metadata'] = meta_back

    def parse_h5_channel_blend_params(self, h5_roi, channel: str):
        """
        Parse the current h5 ROI to extract the blend parameters for one channel

        :param h5_roi: h5 data object containing slots from a previously processed ROI
        :param channel: channel key
        :return: None
        """
        # for blend_key, blend_val in data_h5[roi][channel].items():
        for blend_key, blend_val in h5_roi[channel].items():
            if 'image' not in blend_key:
                if blend_val[()] != b'None':
                    try:
                        data_add = blend_val[()].decode("utf-8")
                    except AttributeError:
                        data_add = str(blend_val[()])
                else:
                    data_add = None
                self.blend_config[channel][blend_key] = data_add

    def check_for_valid_tiff_panel(self, tiff):
        """
        Check if the length of the tiff matches the current imported panel length. Since most tiff files
        will not contain panel metadata (unless ome), the matching heuristic is length

        :param tiff: Instance of a tifffile tiff object
        :return: None
        """
        if not (all(len(value) == len(tiff.pages) for value in list(self.image_dict['metadata'].values()))) or \
                (self.panel_length is not None and self.panel_length != len(tiff.pages)):
            raise PanelMismatchError("One or more ROIs parsed from tiff appear to have"
                                     " different panel lengths. This is currently not supported by rakaia. "
                                     "Refresh your current session to re-import compatible imaging files.")

    def parse_tiff(self, tiff_file, internal_name=None):
        """
        Parse a tiff file. A tiff file should be multiple pages where each page is an array for raw pixel intensities
        for a particular channel.

        :param tiff_file: path to a compatible tiff file.
        :param internal_name: When not using lazy loading, retain the current ROI selection string
        :return: None
        """
        with TiffFile(tiff_file) as tif:
            tiff_path = Path(tiff_file)
            # IMP: if the length of this tiff is not the same as the current metadata, implies that
            # the files have different channels/panels
            # pass if this is the cases
            if len(self.image_dict['metadata']) > 0:
                self.check_for_valid_tiff_panel(tif)
            multi_channel_index = 1
            basename = str(Path(tiff_path).stem)
            roi = f"{basename}{self.delimiter}slide{str(self.slide_index)}{self.delimiter}acq" if \
                internal_name is None else internal_name
            # treat each tiff as its own ROI and increment the acq index for each one
            self.image_dict[roi] = {}
            for page in tif.pages:
                identifier = str("channel_" + str(multi_channel_index))
                # tiff files could be RGB, so convert to greyscale for compatibility
                self.image_dict[roi][identifier] = None if (self.lazy_load or
                        roi_requires_single_marker_load(int(page.shape[0] * page.shape[1]),
                        len(tif.pages))) else convert_rgb_to_greyscale(
                    page.asarray()).astype(set_array_storage_type_from_config(self.array_store_type))
                # add in a generic description for the ROI per tiff file
                if multi_channel_index == 1:
                    self.dataset_information_frame["ROI"].append(str(roi))
                    self.dataset_information_frame["Dimensions"].append(
                        f"{page.asarray().shape[1]}x"
                        f"{page.asarray().shape[0]}")
                    self.dataset_information_frame["Panel"].append(
                        f"{len(tif.pages)} markers")
                multi_channel_index += 1
                self.append_channel_identifier_to_collection(identifier)
                self.append_channel_identifier_to_channel_list(identifier)
                self.append_channel_alias_to_label_list(identifier)

            if len(self.image_dict['metadata']) < 1:
                self.set_hash_metadata(self.metadata_channels, self.metadata_labels)
        self.panel_length = len(tif.pages) if self.panel_length is None else self.panel_length
        self.acq_index += 1

    def check_for_valid_mcd_panel(self, acq, channel_labels):
        """
        Check if the mcd acquisition has a panel length that matches the current imported panel.

        :param acq: Instance of an Acquisition class from `readimc`
        :param channel_labels: List of channel labels from previously parsed acquisitions
        :return: None
        """
        if len(acq.channel_labels) != len(channel_labels) or \
                (self.panel_length is not None and self.panel_length != len(acq.channel_labels)):
            raise PanelMismatchError("One or more ROIs parsed from .mcd appear to have"
                                     " different panel lengths. This is currently not supported by rakaia. "
                                     "Refresh your current session to re-import compatible imaging files.")

    def parse_mcd(self, mcd_filepath):
        """
        Parse a mcd file.

        :param mcd_filepath: path to a compatible mcd file.
        :return: None
        """
        with MCDFile(mcd_filepath) as mcd_file:
            channel_names = None
            channel_labels = None
            slide_index = 0
            acq_index = 0
            for slide in mcd_file.slides:
                for acq in slide.acquisitions:
                    basename = str(Path(mcd_filepath).stem)
                    roi = f"{str(basename)}{self.delimiter}slide{str(slide_index)}" \
                          f"{self.delimiter}{str(acq.description)}_{str(acq.id)}"
                    self.image_dict[roi] = {}
                    if channel_labels is None:
                        channel_labels = acq.channel_labels
                        channel_names = acq.channel_names
                        self.set_hash_metadata(list(channel_names), list(channel_labels))
                    else:
                        # for now, just checking the that the length matches is sufficient in case
                        # there are slight spelling errors between mcds with the same panel
                        self.check_for_valid_mcd_panel(acq, channel_labels)
                    channel_index = 0
                    for channel in acq.channel_names:
                        self.image_dict[roi][channel] = None if (self.lazy_load or
                        roi_requires_single_marker_load(channel, len(acq.channel_labels))) else channel.astype(
                                    set_array_storage_type_from_config(self.array_store_type))
                        self.append_channel_identifier_to_collection(channel_names[channel_index])
                        # add information about the ROI into the description list
                        if channel_index == 0:
                            dim_width = acq.metadata['MaxX'] if 'MaxX' in acq.metadata else "NA"
                            dim_height = acq.metadata['MaxY'] if 'MaxY' in acq.metadata else "NA"

                            self.dataset_information_frame["ROI"].append(str(roi))
                            self.dataset_information_frame["Dimensions"].append(f"{dim_width}x{dim_height}")
                            self.dataset_information_frame["Panel"].append(
                                f"{len(acq.channel_names)} markers")

                        channel_index += 1
                    self.panel_length = len(acq.channel_labels) if self.panel_length is None else self.panel_length
                    acq_index += 1
                slide_index += 1
            mcd_file.close()
        self.experiment_index += 1

    def read_single_roi_from_mcd(self, mcd_filepath, internal_name, roi_name):
        """
        Read a single ROI into the dictionary from a .mcd file.

        :param mcd_filepath: path to a compatible tiff file.
        :param roi_name: When parsing mcd files and not using lazy loading, pass a single ROI name to pull from an mcd.
        :param internal_name: When not using lazy loading, retain the current ROI selection string
        :return: None
        """
        with MCDFile(mcd_filepath) as mcd_file:
            self.image_dict = {internal_name: {}}
            for slide_inside in mcd_file.slides:
                for acq in slide_inside.acquisitions:
                    pattern = f"{str(acq.description)}_{str(acq.id)}"
                    if pattern == roi_name:
                        self.image_dict = self.initialize_empty_mcd_single_read(self.image_dict,
                                        internal_name, list(acq.channel_names))
                        channel_names = acq.channel_names
                        if not roi_requires_single_marker_load(int(int(acq.metadata['MaxX']) * int(acq.metadata['MaxY'])),
                                len(acq.channel_names)):
                            channel_index = 0
                            img = mcd_file.read_acquisition(acq, strict=False)
                            for channel in img:
                                self.image_dict[internal_name][channel_names[channel_index]] = channel.astype(
                                set_array_storage_type_from_config(self.array_store_type))
                                channel_index += 1
                        mcd_file.close()

    @staticmethod
    def initialize_empty_mcd_single_read(image_dict: dict, internal_name: str, channel_list: list):
        """
        Initialize an image dictionary for a single ROI parse prior to checking that it requires single marker
        lazy loading.
        """
        if internal_name and channel_list and internal_name in image_dict:
            for channel in channel_list:
                image_dict[internal_name][channel] = None
        return image_dict


    def check_for_valid_txt_panel(self, txt_channel_names, txt_channel_labels):
        """
        Check if the panel length from a .txt file is compatible with the currently imported panel.

        :param txt_channel_names: List of channel key identifiers internally in the txt file
        :param txt_channel_labels: List of labels to be used in the app display for each channel key
        :return: None
        """
        if not len(self.metadata_channels) == len(txt_channel_names) or \
                not len(self.metadata_labels) == len(txt_channel_labels) or \
                (self.panel_length is not None and self.panel_length != len(txt_channel_names)):
            raise PanelMismatchError("One or more ROIs parsed from .txt appear to have"
                                     " different panel lengths. This is currently not supported by rakaia. "
                                     "Refresh your current session to re-import compatible imaging files.")

    def parse_txt(self, txt_filepath, internal_name=None):
        """
        Parse a compatible txt file. Txt files are often used as backup files for individual ROIs generated from
        mcd, and can be used if the mcd has become corrupted.

        :param txt_filepath: path to a compatible txt file.
        :param internal_name: When not using lazy loading, retain the current ROI selection string
        :return: None
        """
        with TXTFile(txt_filepath) as acq_text_read:
            image_index = 1
            txt_channel_names = acq_text_read.channel_names
            txt_channel_labels = acq_text_read.channel_labels
            # check that the channel names and labels are the same if an upload has already passed
            if len(self.metadata_channels) > 0:
                self.check_for_valid_txt_panel(txt_channel_names, txt_channel_labels)
            basename = str(Path(txt_filepath).stem)
            roi = f"{str(basename)}{self.delimiter}slide{str(self.slide_index)}" \
                  f"{self.delimiter}acq" if internal_name is None else internal_name
            self.image_dict[roi] = {}
            if not self.lazy_load:
                acq = acq_text_read.read_acquisition(strict=False)
            else:
                acq = range(len(txt_channel_names))
            for image in acq:
                image_label = txt_channel_labels[image_index - 1]
                identifier = txt_channel_names[image_index - 1]
                self.image_dict[roi][identifier] = None if self.lazy_load else image.astype(
                    set_array_storage_type_from_config(self.array_store_type))
                if image_index == 1:
                    self.dataset_information_frame["ROI"].append(str(roi))
                    # dimensions cannot be parsed from txt without reading everything into memory
                    dimensions = f"{image.shape[1]}x{image.shape[0]}" if not self.lazy_load else "NA"
                    self.dataset_information_frame["Dimensions"].append(dimensions)
                    self.dataset_information_frame["Panel"].append(
                        f"{len(acq_text_read.channel_names)} markers")
                image_index += 1
                self.append_channel_identifier_to_collection(identifier)
                self.append_channel_identifier_to_channel_list(identifier)
                self.append_channel_alias_to_label_list(image_label)
            if len(self.image_dict['metadata']) < 1:
                self.set_hash_metadata(self.metadata_channels, self.metadata_labels)
            self.panel_length = len(txt_channel_names) if self.panel_length is None else self.panel_length
        self.acq_index += 1

    def parse_h5ad(self, h5ad_filepath):
        """
        Parse an .h5ad filepath. Current technologies that are explicitly supported: 10X Visium, Xenium

        :param h5ad_filepath: Filepath to a spatial dataset with an .h5ad extension
        """
        anndata = ad.read_h5ad(h5ad_filepath)
        if is_spot_based_spatial(anndata) or is_spatial_dataset(anndata):
            basename = str(Path(h5ad_filepath).stem)
            roi = f"{basename}{self.delimiter}slide{str(self.slide_index)}{self.delimiter}acq"
            self.metadata_channels = list(anndata.var_names)
            self.metadata_labels = list(anndata.var_names)
            # get the channel names from the var names
            self.dataset_information_frame["ROI"].append(str(roi))
            grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(anndata)
            self.dataset_information_frame["Dimensions"].append(f"{grid_width}x{grid_height}")
            self.dataset_information_frame["Panel"].append(
                f"{len(list(anndata.var_names))} markers")
            self.image_dict[roi] = {str(marker): None for marker in anndata.var_names}
            self.panel_length = len(list(anndata.var_names))
            self.set_hash_metadata(self.metadata_channels, self.metadata_labels)

    def set_hash_metadata(self, identifiers: list, labels: list):
        """
        Set the image dictionary metadata table and column labels

        :param identifiers: List of keys for each channel
        :param labels: List of aliases/labels for each channel

        :return: None
        """
        self.image_dict['metadata'] = {'Channel Order': range(1, len(identifiers) + 1, 1),
                                       'Channel Name': identifiers,
                                       'Channel Label': labels,
                                       'rakaia Label': labels}
        self.image_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                               'rakaia Label']

    def get_parsed_information(self) -> pd.DataFrame:
        """
        Get a dataframe listing successfully parsed ROIs.

        :return: `pd.DataFrame` listing parsed ROIsw with their dimensions and panel length.
        """
        if isinstance(self.dataset_information_frame, dict) and \
                all(len(col) > 0 for col in self.dataset_information_frame.values()) or \
                isinstance(self.dataset_information_frame, pd.DataFrame) and len(self.dataset_information_frame) > 0:
            return self.dataset_information_frame
        raise NoAcquisitionsParsedError(f"No acquisitions were successfully parsed from the following files: \n"
                                        f"{self.filepaths}. Please review the input files and refresh the session.")

def create_new_blending_dict(uploaded):
    """
    Create a new blending/config dictionary from an uploaded dictionary
    """
    current_blend_dict = {}
    panel_length = None
    for roi in uploaded.keys():
        if "metadata" not in roi:
            if panel_length is None:
                panel_length = len(uploaded[roi].keys())
            if len(uploaded[roi].keys()) != panel_length:
                raise PanelMismatchError("The imported file(s) appear to have different panel lengths. "
                                         "This is currently not supported by rakaia. "
                            "Refresh your current session to re-import compatible imaging files.")
    first_roi = [elem for elem in list(uploaded.keys()) if 'metadata' not in elem][0]
    for channel in uploaded[first_roi].keys():
        current_blend_dict[channel] = {'color': None, 'x_lower_bound': None, 'x_upper_bound': None,
                                       'filter_type': None, 'filter_val': None, 'filter_sigma': None}
        current_blend_dict[channel]['color'] = '#FFFFFF'
    return current_blend_dict if current_blend_dict else dash.no_update


def image_dict_from_lazy_load(dataset_selection: str, session_config: dict,
                              array_store_type: str="float",
                              delimiter: str="+++"):
    """
    Generate an ROI raw array dictionary with an ROI read from a filepath for lazy loading
    """
    # IMP: the copy of the dictionary must be made in case lazy loading isn't required, and all of the data
    # is already contained in the dictionary
    split = split_string_at_pattern(dataset_selection, pattern=delimiter)
    basename, slide, acq_name = split[0], split[1], split[2]
    # get the index of the file from the experiment number in the event that there are multiple uploads
    file_path = None
    for files_uploaded in session_config['uploads']:
        if str(Path(files_uploaded).stem) == basename:
            file_path = files_uploaded
    if file_path is not None:
        upload_dict_new = FileParser(filepaths=[file_path], array_store_type=array_store_type,
                                     lazy_load=False, single_roi_parse=True, internal_name=dataset_selection,
                                     roi_name=acq_name, delimiter=delimiter).image_dict
        return upload_dict_new
    return None

def sparse_array_to_dense(array):
    """
    Return a dense representation of the array if it is sparse, otherwise return as is
    """
    if issparse(array):
        return array.toarray(order='F')
    return array

def convert_between_dense_sparse_array(array, array_type="dense"):
    """
    Convert between dense and sparse arrays. Sparse arrays are stored column wise
    """
    if array_type not in ["dense", "sparse"]:
        raise TypeError("The array type must be either dense or sparse")
    return csc_matrix(sparse_array_to_dense(array)) if array_type == "sparse" else sparse_array_to_dense(array)

def convert_rgb_to_greyscale(array: Union[np.array, np.ndarray]):
    """
    Convert an RGB image to greyscale based on the array shape (number of channels/dimensions).
    """
    if len(array.shape) >= 3:
        return np.array(Image.fromarray(array.astype(np.uint8)).convert('L')).astype(np.float32)
    return array


def populate_alias_dict_from_editable_metadata(metadata: Union[dict, list, pd.DataFrame]):
    """
    Generate a dictionary of channel aliases from the key names based on an editable metadata dataframe
    """
    alias_dict = {}
    for elem in metadata:
        try:
            alias_dict[elem['Channel Name']] = elem['rakaia Label']
        except KeyError:
            try:
                alias_dict[elem['Channel Name']] = elem['Channel Name']
            except KeyError:
                pass
    return alias_dict

def check_blend_dictionary_for_blank_bounds_by_channel(blend_dict: dict, channel_selected: str,
                                                       channel_dict: dict, data_selection: str):
    """
    Check the current blend dictionary for the lower and upper bounds for a specific channel
    If the bounds are None, replace with the default values
    """
    if blend_dict[channel_selected]['x_lower_bound'] in [None, "None", "null"]:
        blend_dict[channel_selected]['x_lower_bound'] = 0
    if blend_dict[channel_selected]['x_upper_bound'] in [None, "None", "null"]:
        blend_dict[channel_selected]['x_upper_bound'] = \
            get_default_channel_upper_bound_by_percentile(
                channel_dict[data_selection][channel_selected])
    return blend_dict

def check_empty_missing_layer_dict(current_layers: Union[dict, None], data_selection: str):
    """
    The layer hash holds the RGB channel arrays for each channel that is processed in an ROI.
    Check if the current layer hash has a hash for the current ROI. If not, create an empty hash
    with the new ROI. This enables the hash to be cleared each time the ROI is changed, to minimize
    the amount of memory used for channel colour arrays
    """
    if current_layers is None or data_selection not in current_layers.keys():
        current_layers = {data_selection: {}}
    return current_layers

def set_current_channels(image_dict: dict, data_selection: str, current_selection: list):
    """
    Set the currently selected channels by verifying that every selection is in the current ROI dictionary.
    Called when switching ROIs with a current blend applied
    """
    if current_selection is not None and len(current_selection) > 0 and \
        data_selection in image_dict and \
            all([elem in image_dict[data_selection].keys() for elem in current_selection]):
        return list(current_selection)
    return []
