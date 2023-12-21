import pandas as pd
import h5py
from pathlib import Path
from tifffile import TiffFile
import os
from ccramic.utils.pixel_level_utils import (
    split_string_at_pattern,
    set_array_storage_type_from_config)
from readimc import MCDFile, TXTFile

def populate_upload_dict(uploaded_files, array_store_type="float"):
    """
    Populate a dictionary based on the uploaded files.
    """
    filenames = [str(x) for x in uploaded_files]
    upload_dict = {}
    unique_image_names = []
    dataset_information = {"ROI": [], "Dimensions": [], "Panel": []}
    if len(filenames) > 0:
        upload_dict['metadata'] = {}
        metadata_channels = []
        metadata_labels = []
        experiment_index = 0
        slide_index = 0
        acq_index = 0
        blend_dict = None
        for upload in filenames:
            # if reading back in data with h5
            if upload.endswith('.h5'):
                blend_dict = upload_dict.copy()
                data_h5 = h5py.File(upload, "r")
                for roi in list(data_h5.keys()):
                    upload_dict[roi] = {}
                    if 'metadata' not in roi:
                        channel_index = 1
                        for channel in data_h5[roi]:
                            try:
                                upload_dict[roi][channel] = data_h5[roi][channel]['image'][()]
                                if channel_index == 1:
                                    dataset_information["ROI"].append(str(roi))
                                    dataset_information["Dimensions"].append(f"{upload_dict[roi][channel].shape[1]}x"
                                    f"{upload_dict[roi][channel].shape[0]}")
                                    dataset_information["Panel"].append(f"{len(data_h5[roi].keys())} markers")
                            except KeyError:
                                pass
                            if channel not in unique_image_names:
                                unique_image_names.append(channel)
                            blend_dict[channel] = {}
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
                                    blend_dict[channel][blend_key] = data_add
                    else:
                        meta_back = pd.DataFrame(data_h5['metadata'])
                        for col in meta_back.columns:
                            meta_back[col] = meta_back[col].str.decode("utf-8")
                        try:
                            meta_back.columns = [i.decode("utf-8") for i in data_h5['metadata_columns']]
                        except KeyError:
                            pass
                        upload_dict['metadata'] = meta_back
            else:
                # if tiffs are uploaded, treat as one slide and one acquisition
                if upload.endswith('.tiff') or upload.endswith('.tif'):
                    try:
                        with TiffFile(upload) as tif:
                            tiff_path = Path(upload)
                            # IMP: if the length of this tiff is not the same as the current metadata, implies that
                            # the files have different channels/panels
                            # pass if this is the cases
                            if len(upload_dict['metadata']) > 0:
                                assert all(len(value) == len(tif.pages) for value in \
                                           list(upload_dict['metadata'].values()))

                            file_name, file_extension = os.path.splitext(tiff_path)
                            # set different image labels based on the basename of the file (ome.tiff vs .tiff)
                            # if "ome" in upload:
                            #     basename = str(os.path.basename(tiff_path)).split(".ome" + file_extension)[0]
                            # else:
                            #     basename = str(os.path.basename(tiff_path)).split(file_extension)[0]
                            multi_channel_index = 1
                            basename = str(Path(upload).stem)
                            roi = f"{basename}+++slide{str(slide_index)}+++acq{str(acq_index)}"
                            # treat each tiff as a its own ROI and increment the acq index for each one
                            upload_dict[roi] = {}
                            for page in tif.pages:
                                # identifier = str(basename) + str("_channel_" + f"{multi_channel_index}") if \
                                #     len(tif.pages) > 1 else str(basename)
                                identifier = str("channel_" + str(multi_channel_index))
                                upload_dict[roi][identifier] = page.asarray().astype(
                                    set_array_storage_type_from_config(array_store_type))
                                # add in a generic description for the ROI per tiff file
                                if multi_channel_index == 1:
                                    dataset_information["ROI"].append(str(roi))
                                    dataset_information["Dimensions"].append(
                                        f"{page.asarray().shape[1]}x" \
                                                  f"{page.asarray().shape[0]}")
                                    dataset_information["Panel"].append(
                                        f"{len(tif.pages)} markers")
                                multi_channel_index += 1
                                if identifier not in metadata_channels:
                                    metadata_channels.append(identifier)
                                if identifier not in metadata_labels:
                                    metadata_labels.append(identifier)
                                if identifier not in unique_image_names:
                                    unique_image_names.append(identifier)

                            if len(upload_dict['metadata']) < 1:
                                upload_dict['metadata'] = {'Channel Order': range(1, len(metadata_channels) + 1, 1),
                                               'Channel Name': metadata_channels,
                                               'Channel Label': metadata_labels,
                                               'ccramic Label': metadata_labels}
                                upload_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                                   'ccramic Label']
                        acq_index += 1
                    except AssertionError:
                        pass
                if upload.endswith('.mcd'):
                    with MCDFile(upload) as mcd_file:
                        channel_names = None
                        channel_labels = None
                        slide_index = 0
                        acq_index = 0
                        for slide in mcd_file.slides:
                            for acq in slide.acquisitions:
                                basename = str(Path(upload).stem)
                                roi = f"{str(basename)}+++slide{str(slide_index)}" \
                                      f"+++{str(acq.description)}"
                                upload_dict[roi] = {}
                                if channel_labels is None:
                                    channel_labels = acq.channel_labels
                                    channel_names = acq.channel_names
                                    upload_dict['metadata'] = {'Channel Order': range(1, len(channel_names) + 1, 1),
                                                               'Channel Name': channel_names,
                                                               'Channel Label': channel_labels,
                                                               'ccramic Label': channel_labels}
                                    upload_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                                       'ccramic Label']
                                else:
                                    #TODO: establish within an MCD if the channel names should be exactly the same
                                    # or if the length is sufficient
                                    # i.e. how to handle minor spelling mistakes
                                    # assert all(label in acq.channel_labels for label in channel_labels)
                                    # assert all(name in acq.channel_names for name in channel_names)
                                    assert len(acq.channel_labels) == len(channel_labels)
                                # img = mcd_file.read_acquisition(acq)
                                channel_index = 0
                                for channel in acq.channel_names:
                                    # TODO: implement lazy loading (only read in images in
                                    #  ROI selection from the dropdown)
                                    upload_dict[roi][channel] = None
                                    if channel_names[channel_index] not in unique_image_names:
                                        unique_image_names.append(channel_names[channel_index])
                                    # add information about the ROI into the description list
                                    if channel_index == 0:
                                        dim_width = acq.metadata['MaxX'] if 'MaxX' in acq.metadata else "NA"
                                        dim_height = acq.metadata['MaxY'] if 'MaxY' in acq.metadata else "NA"

                                        dataset_information["ROI"].append(str(roi))
                                        dataset_information["Dimensions"].append(f"{dim_width}x{dim_height}")
                                        dataset_information["Panel"].append(
                                            f"{len(acq.channel_names)} markers")

                                    channel_index += 1
                                acq_index += 1
                            slide_index += 1
                    experiment_index += 1
                if upload.endswith('.txt'):
                    try:
                        with TXTFile(upload) as acq_text_read:
                            acq = acq_text_read.read_acquisition()
                            image_index = 1
                            txt_channel_names = acq_text_read.channel_names
                            txt_channel_labels = acq_text_read.channel_labels
                            # assert that the channel names and labels are the same if an upload has already passed
                            if len(metadata_channels) > 0:
                                assert len(metadata_channels) == len(txt_channel_names)
                                assert all([elem in txt_channel_names for elem in metadata_channels])
                            if len(metadata_labels) > 0:
                                assert len(metadata_labels) == len(txt_channel_labels)
                                assert all([elem in txt_channel_labels for elem in metadata_labels])
                            basename = str(Path(upload).stem)
                            roi = f"{str(basename)}+++slide{str(slide_index)}" \
                                  f"+++{str(acq_index)}"
                            upload_dict[roi] = {}
                            for image in acq:
                                image_label = txt_channel_labels[image_index - 1]
                                identifier = txt_channel_names[image_index - 1]
                                upload_dict[roi][identifier] = image.astype(
                                    set_array_storage_type_from_config(array_store_type))
                                if image_index == 1:
                                    dataset_information["ROI"].append(str(roi))
                                    dataset_information["Dimensions"].append(f"{image.shape[1]}x{image.shape[0]}")
                                    dataset_information["Panel"].append(
                                        f"{len(acq_text_read.channel_names)} markers")
                                image_index += 1
                                if identifier not in metadata_channels:
                                    metadata_channels.append(identifier)
                                if image_label not in metadata_labels:
                                    metadata_labels.append(image_label)
                                if identifier not in unique_image_names:
                                    unique_image_names.append(identifier)
                            if len(upload_dict['metadata']) < 1:
                                upload_dict['metadata'] = {'Channel Order': range(1, len(metadata_channels) + 1, 1),
                                                           'Channel Name': metadata_channels,
                                                           'Channel Label': metadata_labels,
                                                           'ccramic Label': metadata_labels}
                                upload_dict['metadata_columns'] = ['Channel Order', 'Channel Name', 'Channel Label',
                                                                   'ccramic Label']
                        acq_index += 1
                    except (OSError, AssertionError):
                        pass
        return upload_dict, blend_dict, unique_image_names, dataset_information
    else:
        return None

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
            else:
                assert len(uploaded[roi].keys()) == panel_length
            # assert that all of the rois have the same length to use the same panel for all
    first_roi = [elem for elem in list(uploaded.keys()) if 'metadata' not in elem][0]
    for channel in uploaded[first_roi].keys():
        current_blend_dict[channel] = {'color': None, 'x_lower_bound': None, 'x_upper_bound': None,
                                       'filter_type': None, 'filter_val': None, 'filter_sigma': None}
        current_blend_dict[channel]['color'] = '#FFFFFF'
    return current_blend_dict


def populate_upload_dict_by_roi(upload_dict, dataset_selection, session_config, array_store_type="float"):
    """
    Populate an existing upload dictionary with an ROI read from a filepath for lazy loading
    """
    #IMP: the copy of the dictionary must be made in case lazy loading isn't required, and all of the data
    # is already contained in the dictionary
    try:
        split = split_string_at_pattern(dataset_selection)
        basename, slide, acq_name = split[0], split[1], split[2]
        # get the index of the file from the experiment number in the event that there are multiple uploads
        file_path = None
        for files_uploaded in session_config['uploads']:
            if str(Path(files_uploaded).stem) == basename:
                file_path = files_uploaded
        if file_path is not None:
            assert file_path.endswith('.mcd')
            with MCDFile(file_path) as mcd_file:
                upload_dict = {dataset_selection: {}}
                for slide_inside in mcd_file.slides:
                    for acq in slide_inside.acquisitions:
                        if acq.description == acq_name:
                            channel_names = acq.channel_names
                            channel_index = 0
                            img = mcd_file.read_acquisition(acq)
                            for channel in img:
                                upload_dict[dataset_selection][channel_names[channel_index]] = channel.astype(
                                    set_array_storage_type_from_config(array_store_type))
                                channel_index += 1
        return upload_dict
    except (KeyError, AssertionError, AttributeError):
        return upload_dict
