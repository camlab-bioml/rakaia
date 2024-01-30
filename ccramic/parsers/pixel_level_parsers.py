import h5py
from pathlib import Path
from tifffile import TiffFile
from ccramic.utils.pixel_level_utils import (
    split_string_at_pattern,
    set_array_storage_type_from_config)
from readimc import MCDFile, TXTFile
from scipy.sparse import issparse, csc_matrix
from ccramic.utils.alert import PanelMismatchError

class FileParser:
    """
    Parses a list of filepaths into a dictionary of image arrays, grouped by region (ROI) identifiers
    When using lazy loading, the dictionary will be created as a placeholder and a dataframe of ROI information
    will be created, but the dictionary will contain None values in place of image arrays. When lazy loading is
    turned off, greyscale image arrays are read into the dictionary slots with the numpy array type specified
    in `array_store_type`
    """
    def __init__(self, filepaths: list, array_store_type="float", lazy_load=True,
                 single_roi_parse=True, roi_name=None, internal_name=None, delimiter="+++"):
        if array_store_type not in ["float", "int"]:
            raise TypeError("The array stored type must be one of float or int")
        self.filepaths = [str(x) for x in filepaths]
        self.array_store_type = array_store_type
        self.image_dict = {}
        self.unique_image_names = []
        self.dataset_information_frame = {"ROI": [], "Dimensions": [], "Panel": []}
        self.lazy_load = lazy_load
        self.delimiter = delimiter
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
            for upload in self.filepaths:
                # IMP: split reading a single mcd ROI from the entire mcd, as mcds can contain multiple ROIs
                # this is currently unique to mcds: all other files have one ROI per file
                if upload.endswith('.mcd') and not lazy_load and single_roi_parse and \
                        None not in (roi_name, internal_name):
                   self.read_single_roi_from_mcd(upload, self.internal_name, self.roi_name)
                elif upload.endswith('.mcd'):
                    self.parse_mcd(upload)
                elif upload.endswith('.h5'):
                    self.parse_h5(upload)
                elif upload.endswith('.tiff') or upload.endswith('.tif'):
                    self.parse_tiff(upload, internal_name=internal_name)
                elif upload.endswith('.txt'):
                    try:
                        self.parse_txt(upload, internal_name=internal_name)
                    except (OSError, AssertionError):
                        pass
                else:
                    raise TypeError(f"{upload} is not one of the supported image filetypes:\n"
                                    ".mcd, .tiff, .txt, or .h5")

    def parse_h5(self, h5py_file):
        data_h5 = h5py.File(h5py_file, "r")
        self.blend_config = {}
        for roi in list(data_h5.keys()):
            self.image_dict[roi] = {}
            if 'metadata' not in roi:
                channel_index = 1
                for channel in data_h5[roi]:
                    try:
                        self.image_dict[roi][channel] = data_h5[roi][channel]['image'][()]
                        if channel_index == 1:
                            self.dataset_information_frame["ROI"].append(str(roi))
                            self.dataset_information_frame["Dimensions"].append(
                                f"{self.image_dict[roi][channel].shape[1]}x"
                                f"{self.image_dict[roi][channel].shape[0]}")
                            self.dataset_information_frame["Panel"].append(f"{len(data_h5[roi].keys())} markers")
                    except KeyError:
                        pass
                    if channel not in self.unique_image_names:
                        self.unique_image_names.append(channel)
                    self.blend_config[channel] = {}
                    channel_index += 1
                    for blend_key, blend_val in data_h5[roi][channel].items():
                        if 'image' not in blend_key:
                            if blend_val[()] != b'None':
                                try:
                                    data_add = blend_val[()].decode("utf-8")
                                except AttributeError:
                                    data_add = str(blend_val[()])
                            else:
                                data_add = None
                            self.blend_config[channel][blend_key] = data_add

    def parse_tiff(self, tiff_file, internal_name=None):
        with TiffFile(tiff_file) as tif:
            tiff_path = Path(tiff_file)
            # IMP: if the length of this tiff is not the same as the current metadata, implies that
            # the files have different channels/panels
            # pass if this is the cases
            if len(self.image_dict['metadata']) > 0:
                if not all(len(value) == len(tif.pages) for value in list(self.image_dict['metadata'].values())):
                    raise PanelMismatchError("The tiff file(s) appear to have different panels"
                                             ". This is currently not supported by ccramic.")
            # file_name, file_extension = os.path.splitext(tiff_path)
            # set different image labels based on the basename of the file (ome.tiff vs .tiff)
            # if "ome" in upload:
            #     basename = str(os.path.basename(tiff_path)).split(".ome" + file_extension)[0]
            # else:
            #     basename = str(os.path.basename(tiff_path)).split(file_extension)[0]
            multi_channel_index = 1
            basename = str(Path(tiff_path).stem)
            roi = f"{basename}{self.delimiter}slide{str(self.slide_index)}{self.delimiter}acq{str(self.acq_index)}" if \
                internal_name is None else internal_name
            # treat each tiff as a its own ROI and increment the acq index for each one
            self.image_dict[roi] = {}
            for page in tif.pages:
                # identifier = str(basename) + str("_channel_" + f"{multi_channel_index}") if \
                #     len(tif.pages) > 1 else str(basename)
                identifier = str("channel_" + str(multi_channel_index))
                self.image_dict[roi][identifier] = None if self.lazy_load else page.asarray().astype(
                    set_array_storage_type_from_config(self.array_store_type))
                # add in a generic description for the ROI per tiff file
                if multi_channel_index == 1:
                    self.dataset_information_frame["ROI"].append(str(roi))
                    self.dataset_information_frame["Dimensions"].append(
                        f"{page.asarray().shape[1]}x" \
                        f"{page.asarray().shape[0]}")
                    self.dataset_information_frame["Panel"].append(
                        f"{len(tif.pages)} markers")
                multi_channel_index += 1
                if identifier not in self.metadata_channels:
                    self.metadata_channels.append(identifier)
                if identifier not in self.metadata_labels:
                    self.metadata_labels.append(identifier)
                if identifier not in self.unique_image_names:
                    self.unique_image_names.append(identifier)

            if len(self.image_dict['metadata']) < 1:
                self.image_dict['metadata'] = {'Channel Order': range(1, len(self.metadata_channels) + 1, 1),
                                           'Channel Name': self.metadata_channels,
                                           'Channel Label': self.metadata_labels,
                                           'ccramic Label': self.metadata_labels}
                self.image_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                   'ccramic Label']
        self.acq_index += 1

    def parse_mcd(self, mcd_filepath):
        with MCDFile(mcd_filepath) as mcd_file:
            channel_names = None
            channel_labels = None
            slide_index = 0
            acq_index = 0
            for slide in mcd_file.slides:
                for acq in slide.acquisitions:
                    basename = str(Path(mcd_filepath).stem)
                    roi = f"{str(basename)}{self.delimiter}slide{str(slide_index)}" \
                          f"{self.delimiter}{str(acq.description)}"
                    self.image_dict[roi] = {}
                    if channel_labels is None:
                        channel_labels = acq.channel_labels
                        channel_names = acq.channel_names
                        self.image_dict['metadata'] = {'Channel Order': range(1, len(channel_names) + 1, 1),
                                                   'Channel Name': channel_names,
                                                   'Channel Label': channel_labels,
                                                   'ccramic Label': channel_labels}
                        self.image_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                           'ccramic Label']
                    else:
                        # TODO: establish within an MCD if the channel names should be exactly the same
                        # or if the length is sufficient
                        # i.e. how to handle minor spelling mistakes
                        # assert all(label in acq.channel_labels for label in channel_labels)
                        # assert all(name in acq.channel_names for name in channel_names)
                        # assert len(acq.channel_labels) == len(channel_labels)
                        if len(acq.channel_labels) != len(channel_labels):
                            raise PanelMismatchError("The mcd file appears that have ROIs with different"
                                                     "panels. This is currently not supported by ccramic.")
                    # img = mcd_file.read_acquisition(acq)
                    channel_index = 0
                    for channel in acq.channel_names:
                        self.image_dict[roi][channel] = None if self.lazy_load else channel.astype(
                                    set_array_storage_type_from_config(self.array_store_type))
                        if channel_names[channel_index] not in self.unique_image_names:
                            self.unique_image_names.append(channel_names[channel_index])
                        # add information about the ROI into the description list
                        if channel_index == 0:
                            dim_width = acq.metadata['MaxX'] if 'MaxX' in acq.metadata else "NA"
                            dim_height = acq.metadata['MaxY'] if 'MaxY' in acq.metadata else "NA"

                            self.dataset_information_frame["ROI"].append(str(roi))
                            self.dataset_information_frame["Dimensions"].append(f"{dim_width}x{dim_height}")
                            self.dataset_information_frame["Panel"].append(
                                f"{len(acq.channel_names)} markers")

                        channel_index += 1
                    acq_index += 1
                slide_index += 1
        self.experiment_index += 1

    def read_single_roi_from_mcd(self, mcd_filepath, internal_name, roi_name):
        """
        Read a single ROI into the dictionary from an mcd file
        the internal name is that string representation of the
        """
        with MCDFile(mcd_filepath) as mcd_file:
            self.image_dict = {internal_name: {}}
            for slide_inside in mcd_file.slides:
                for acq in slide_inside.acquisitions:
                    if acq.description == roi_name:
                        channel_names = acq.channel_names
                        channel_index = 0
                        img = mcd_file.read_acquisition(acq)
                        for channel in img:
                            self.image_dict[internal_name][channel_names[channel_index]] = channel.astype(
                                set_array_storage_type_from_config(self.array_store_type))
                            channel_index += 1

    def parse_txt(self, txt_filepath, internal_name=None):
        with TXTFile(txt_filepath) as acq_text_read:
            image_index = 1
            txt_channel_names = acq_text_read.channel_names
            txt_channel_labels = acq_text_read.channel_labels
            # assert that the channel names and labels are the same if an upload has already passed
            # TODO: add custom exception rule here for mismatched panels
            if len(self.metadata_channels) > 0:
                if not len(self.metadata_channels) == len(txt_channel_names) or \
                        not len(self.metadata_labels) == len(txt_channel_labels):
                    raise PanelMismatchError("The txt file(s) appear to have different panels"
                                             ". This is currently not supported by ccramic.")
                # assert len(metadata_channels) == len(txt_channel_names)
            #     # assert all([elem in txt_channel_names for elem in metadata_channels])
            # if len(metadata_labels) > 0:
            #     assert len(metadata_labels) == len(txt_channel_labels)
            #     assert all([elem in txt_channel_labels for elem in metadata_labels])
            basename = str(Path(txt_filepath).stem)
            roi = f"{str(basename)}{self.delimiter}slide{str(self.slide_index)}" \
                  f"{self.delimiter}{str(self.acq_index)}" if internal_name is None else internal_name
            self.image_dict[roi] = {}
            # TODO: only read the acquisition if lazy loading is off
            if not self.lazy_load:
                acq = acq_text_read.read_acquisition()
            else:
                acq = range(len(txt_channel_names))
            for image in acq:
                image_label = txt_channel_labels[image_index - 1]
                identifier = txt_channel_names[image_index - 1]
                # TODO: potentially implement lazy loading here for .txt files
                self.image_dict[roi][identifier] = None if self.lazy_load else image.astype(
                    set_array_storage_type_from_config(self.array_store_type))
                if image_index == 1:
                    self.dataset_information_frame["ROI"].append(str(roi))
                    # TODO: see if you can get dimensions without reading the entire acq, as this is time consuming
                    dimensions = f"{image.shape[1]}x{image.shape[0]}" if not self.lazy_load else "NA"
                    self.dataset_information_frame["Dimensions"].append(dimensions)
                    self.dataset_information_frame["Panel"].append(
                        f"{len(acq_text_read.channel_names)} markers")
                image_index += 1
                if identifier not in self.metadata_channels:
                    self.metadata_channels.append(identifier)
                if image_label not in self.metadata_labels:
                    self.metadata_labels.append(image_label)
                if identifier not in self.unique_image_names:
                    self.unique_image_names.append(identifier)
            if len(self.image_dict['metadata']) < 1:
                self.image_dict['metadata'] = {'Channel Order': range(1, len(self.metadata_channels) + 1, 1),
                                           'Channel Name': self.metadata_channels,
                                           'Channel Label': self.metadata_labels,
                                           'ccramic Label': self.metadata_labels}
                self.image_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                   'ccramic Label']
        self.acq_index += 1

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
                raise PanelMismatchError("The imported file(s) appear to have different panels"
                                         ". This is currently not supported by ccramic.")
            # assert that all of the rois have the same length to use the same panel for all
    first_roi = [elem for elem in list(uploaded.keys()) if 'metadata' not in elem][0]
    for channel in uploaded[first_roi].keys():
        current_blend_dict[channel] = {'color': None, 'x_lower_bound': None, 'x_upper_bound': None,
                                       'filter_type': None, 'filter_val': None, 'filter_sigma': None}
        current_blend_dict[channel]['color'] = '#FFFFFF'
    return current_blend_dict


def populate_image_dict_from_lazy_load(upload_dict, dataset_selection, session_config, array_store_type="float",
                                       delimiter="+++"):
    """
    Populate an existing upload dictionary with an ROI read from a filepath for lazy loading
    """
    #IMP: the copy of the dictionary must be made in case lazy loading isn't required, and all of the data
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
    return upload_dict

def sparse_array_to_dense(array):
    """
    Return a dense representation of the array if it is sparse, otherwise return as is
    """
    if issparse(array):
        return array.toarray(order='F')
    else:
        return array

def convert_between_dense_sparse_array(array, array_type="dense"):
    """
    Convert between dense and sparse arrays. Sparse arrays are stored column wise
    """
    if array_type not in ["dense", "sparse"]:
        raise TypeError("The array type must be either dense or sparse")
    return csc_matrix(sparse_array_to_dense(array)) if array_type == "sparse" else sparse_array_to_dense(array)
