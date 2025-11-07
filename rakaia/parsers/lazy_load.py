"""Module defining functions and classes for single marker array lazy loading from disk
into memory
"""

import os
from functools import partial
from typing import Union
from readimc import MCDFile
from readimc.data.acquisition import Acquisition
from tifffile import TiffFile
from rakaia.utils.pixel import (
    split_string_at_pattern,
    set_array_storage_type_from_config,
    reshape_chan_first)
from rakaia.parsers.spatial import (
    check_spatial_array_multi_channel,
    spatial_canvas_dimensions)


class SingleMarkerLazyLoaderExtensions:
    """
    Defines the file type extensions for datasets that are supported for single marker lazy loading
    """
    extensions = ['.h5ad', '.mcd', '.tiff', '.tif', '.TIF', '.TIFF']


def parse_files_for_lazy_loading(uploads: Union[list, dict], data_selection: str, delimiter: str= "+++"):
    """
    Parse the current upload list for extensions that support single marker lazy loading
    """
    uploads = uploads['uploads'] if isinstance(uploads, dict) and 'uploads' in uploads else uploads
    exp, slide, acq =  split_string_at_pattern(data_selection, delimiter)
    for upload in uploads:
        if any(upload.endswith(ext) for ext in SingleMarkerLazyLoaderExtensions.extensions) and exp in upload:
            return upload
    return None

class SingleMarkerLazyLoader:
    """
    Provides an option to load a single channel/marker from a supported file upload. Single marker
    loading by ROI is determined by both the dimensions of the region and the number of markers

    :param image_dict: Current session dictionary of raw ROI images
    :param data_selection: String representation of the currently loaded ROI
    :param session_uploads: Dictionary containing a list of filepath uploads
    :param spot_size: If loading spatial datasets, what custom spot size to use
    :param delimiter: String used to split the dataset selection string into file, slide, and acquisition identifiers

    :return: None
    """
    # defines the file types that are currently supported for single channel lazy loading
    MATCHES = {".mcd": "mcd", ".tiff": "tiff", ".tif": "tiff", ".txt": "txt", ".h5": "h5",
               ".h5ad": "h5ad", ".TIF": "tiff", ".TIFF": "tiff"}

    def __init__(self, image_dict: dict, data_selection: str,
                session_uploads: dict,
                channels_selected: Union[str, list],
                spot_size: Union[int,float]=55,
                delimiter: str="+++",
                array_store_type: str="float"):
        self.uploads = session_uploads
        self.image_dict = image_dict
        self.delimiter = delimiter
        self.array_store_type = array_store_type
        self.exp, self.slide, self.acq = None, None, None
        self.width, self.height, self.x_min, self.y_min = None, None, 0, 0
        self.h5ad = partial(self.parse_h5ad)
        self.mcd = partial(self.parse_mcd)
        self.tiff = partial(self.parse_tiff)
        if data_selection and delimiter and session_uploads:
            self.data_selection = data_selection
            self.exp, self.slide, self.acq = split_string_at_pattern(data_selection, delimiter)
            self.channel_selection = [channels_selected] if isinstance(channels_selected, str) else (
                channels_selected)
            self.spot_size = spot_size
            file_to_parse = parse_files_for_lazy_loading(self.uploads, self.data_selection, self.delimiter)
            if file_to_parse:
                filename, file_extension = os.path.splitext(file_to_parse)
                getattr(self, self.MATCHES[file_extension])(file_to_parse)

    def parse_h5ad(self, h5ad_filepath):
        """
        Parse an .h5ad spatial dataset for single marker lazy loading

        :param h5ad_filepath: Filepath to a supported .h5ad spatial dataset

        :return: None
        """
        self.width, self.height, self.x_min, self.y_min = spatial_canvas_dimensions(h5ad_filepath)
        if self.channel_selection:
            self.image_dict = check_spatial_array_multi_channel(self.image_dict, self.data_selection,
                            h5ad_filepath, self.channel_selection, self.spot_size, self.array_store_type)

    def set_mcd_acq_region_dims(self, acq: Acquisition):
        """
        Set the dimensions of an acquisition region parsed from a .mcd file

        :param acq: Acquisition object containing the multi-channel array for a specific ROI

        :return: None
        """
        self.width = int(acq.metadata['MaxX']) if 'MaxX' in acq.metadata else None
        self.height = int(acq.metadata['MaxY']) if 'MaxY' in acq.metadata else None

    def parse_mcd(self, mcd_filepath):
        """
        Parse a mcd IMC file for single marker lazy loading

        :param mcd_filepath: Filepath to a mcd file

        :return: None
        """
        with (MCDFile(mcd_filepath) as mcd_file):
            for slide in mcd_file.slides:
                for acq in slide.acquisitions:
                    pattern = f"{str(acq.description)}_{str(acq.id)}"
                    if pattern == self.acq:
                        self.set_mcd_acq_region_dims(acq)
                        if self.channel_selection:
                            chan_indices = [int(acq.channel_names.index(chan)) for chan in self.channel_selection]
                            img_lazy = mcd_file.read_acquisition(acq, strict=False, channels=chan_indices)
                            for selection, img in zip(self.channel_selection, img_lazy):
                                if not self.image_dict[self.data_selection] or \
                                        self.image_dict[self.data_selection][selection] is None:
                                    self.image_dict[self.data_selection][selection] = reshape_chan_first(img).astype(
                                        set_array_storage_type_from_config(self.array_store_type))

    def set_tiff_region_dims(self, tiff: TiffFile):
        """
        Set the dimensions of a region parsed from tiff

        :param tiff: TiffFile object containing the channel pages for a specific ROI

        :return: None
        """
        if len(tiff.pages) > 0:
            self.height, self.width = int(tiff.pages[0].shape[0]), int(tiff.pages[0].shape[1])


    def parse_tiff(self, tiff_filepath):
        """
        Parse a tiff file for single marker lazy loading

        :param tiff_filepath: Filepath to a mcd file

        :return: None
        """
        with TiffFile(tiff_filepath) as tif:
            self.set_tiff_region_dims(tif)
            for page in tif.pages:
                chan_identifier = f"channel_{int(page.index + 1)}"
                if self.channel_selection and chan_identifier in self.channel_selection and \
                        (not self.image_dict[self.data_selection] or
                        self.image_dict[self.data_selection][chan_identifier] is None):
                    self.image_dict[self.data_selection][chan_identifier] = page.asarray().astype(
                    set_array_storage_type_from_config(self.array_store_type))

    def get_image_dict(self):
        """

        :return: The updated image dictionary for the current ROI
        """
        return self.image_dict

    def get_region_dim(self):
        """

        :return: A tuple of region dimensions and axis offsets for the current ROI
        """
        return self.width, self.height, self.x_min, self.y_min
