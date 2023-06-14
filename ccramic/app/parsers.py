import pandas as pd
import h5py
from pathlib import Path
from tifffile import TiffFile
import os
from .utils import *
from readimc import MCDFile, TXTFile


def populate_upload_dict(uploaded_files):
    """
    Populate a nested dictionary based on the uploaded files.
    """
    filenames = [str(x) for x in uploaded_files]
    upload_dict = {}
    unique_image_names = []
    dataset_information = []
    if len(filenames) > 0:
        upload_dict['metadata'] = {}
        metadata_channels = []
        metadata_labels = []
        experiment_index = 0
        slide_index = 0
        acq_index = 0
        upload_dict["experiment" + str(experiment_index)] = {}
        upload_dict["experiment" + str(experiment_index)]["slide" + str(0)] = {}
        upload_dict["experiment" + str(experiment_index)]["slide" + str(0)]["acq" + str(0)] = {}
        blend_dict = None
        for upload in filenames:
            # if reading back in data with h5
            if upload.endswith('.h5'):
                blend_dict = upload_dict.copy()
                data_h5 = h5py.File(upload, "r")
                for exp in list(data_h5.keys()):
                    upload_dict[exp] = {}
                    blend_dict[exp] = {}
                    if 'metadata' not in exp:
                        for slide in data_h5[exp].keys():
                            upload_dict[exp][slide] = {}
                            blend_dict[exp][slide] = {}
                        for acq in data_h5[exp][slide].keys():
                            upload_dict[exp][slide][acq] = {}
                            blend_dict[exp][slide][acq] = {}
                            channel_index = 1
                            for channel in data_h5[exp][slide][acq]:
                                try:
                                    upload_dict[exp][slide][acq][channel] = data_h5[exp][slide][acq][channel]['image'][()]
                                    if channel_index == 1:
                                        description = f"{acq}, Dimensions: {upload_dict[exp][slide][acq][channel].shape[1]}x" \
                                                      f"{upload_dict[exp][slide][acq][channel].shape[0]}, " \
                                                      f"Panel: {len(data_h5[exp][slide][acq].keys())} markers"
                                        dataset_information.append(description)
                                except KeyError:
                                    pass
                                if channel not in unique_image_names:
                                    unique_image_names.append(channel)
                                blend_dict[exp][slide][acq][channel] = {}
                                channel_index += 1
                                for blend_key, blend_val in data_h5[exp][slide][acq][channel].items():
                                    if 'image' not in blend_key:
                                        if blend_val[()] != b'None':
                                            try:
                                                data_add = blend_val[()].decode("utf-8")
                                            except AttributeError:
                                                data_add = str(blend_val[()])
                                        else:
                                            data_add = None
                                        blend_dict[exp][slide][acq][channel][blend_key] = data_add
                    else:
                        meta_back = pd.DataFrame(data_h5['metadata'])
                        for col in meta_back.columns:
                            meta_back[col] = meta_back[col].str.decode("utf-8")
                        try:
                            meta_back.columns = [i.decode("utf-8") for i in data_h5['metadata_columns']]
                        except KeyError:
                            pass
                        upload_dict[exp] = meta_back
            else:
                # if tiffs are uploaded, treat as one slide and one acquisition
                if upload.endswith('.tiff') or upload.endswith('.tif'):
                    try:
                        with TiffFile(upload) as tif:
                            tiff_path = Path(upload)
                            # IMP: if the length of this tiff is not the same as the current metadata, implies that
                            # the files have different channels/panels
                            # pass if this is the case

                            if len(upload_dict['metadata']) > 0:
                                assert all(len(value) == len(tif.pages) for value in \
                                           list(upload_dict['metadata'].values()))

                            file__name, file_extension = os.path.splitext(tiff_path)
                            # set different image labels based on the basename of the file (ome.tiff vs .tiff)
                            # if "ome" in upload:
                            #     basename = str(os.path.basename(tiff_path)).split(".ome" + file_extension)[0]
                            # else:
                            #     basename = str(os.path.basename(tiff_path)).split(file_extension)[0]
                            multi_channel_index = 1
                            # treat each tiff as a its own ROI and increment the acq index for each one
                            upload_dict["experiment" + str(experiment_index)]["slide" + \
                                                                          str(slide_index)]["acq" + \
                                                                                            str(acq_index)] = {}
                            for page in tif.pages:
                                # identifier = str(basename) + str("_channel_" + f"{multi_channel_index}") if \
                                #     len(tif.pages) > 1 else str(basename)
                                identifier = str("channel_" + str(multi_channel_index))
                                upload_dict["experiment" + str(experiment_index)]["slide" + \
                                                                              str(slide_index)]["acq" + \
                                                                                                str(acq_index)][
                                    identifier] = convert_to_below_255(page.asarray())
                                # add in a generic description for the ROI per tiff file
                                if multi_channel_index == 1:
                                    description = f"{'acq' + str(acq_index)}, Dimensions: {page.asarray().shape[1]}x" \
                                                  f"{page.asarray().shape[0]}, Panel: {len(tif.pages)} markers"
                                    dataset_information.append(description)
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
                    upload_dict["experiment" + str(experiment_index)] = {}
                    with MCDFile(upload) as mcd_file:
                        channel_names = None
                        channel_labels = None
                        slide_index = 0
                        acq_index = 0
                        for slide in mcd_file.slides:
                            upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)] = {}
                            # acq_index = 0
                            for acq in slide.acquisitions:
                                upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)][
                                    str(acq.description)] = {}
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
                                    assert all(label in acq.channel_labels for label in channel_labels)
                                    assert all(name in acq.channel_names for name in channel_names)
                                img = mcd_file.read_acquisition(acq)
                                channel_index = 0
                                for channel in img:
                                    # TODO: implement lazy loading (only read in images in
                                    #  ROI selection from the dropdown)
                                    upload_dict["experiment" + str(experiment_index)]["slide" +
                                                                                      str(slide_index)][
                                                                                      str(acq.description)][
                                        channel_names[channel_index]] = None
                                    if channel_names[channel_index] not in unique_image_names:
                                        unique_image_names.append(channel_names[channel_index])
                                    # add information about the ROI into the description list
                                    if channel_index == 0:
                                        description = f"{acq.description}, Dimensions: {channel.shape[1]}x" \
                                                      f"{channel.shape[0]}, Panel: {len(acq.channel_names)} markers"
                                        dataset_information.append(description)
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
                            upload_dict["experiment" + str(experiment_index)]["slide" + \
                                                                              str(slide_index)]["acq" + \
                                                                                                str(acq_index)] = {}
                            for image in acq:
                                image_label = txt_channel_labels[image_index - 1]
                                identifier = txt_channel_names[image_index - 1]
                                upload_dict["experiment" + str(experiment_index)]["slide" + \
                                                                                  str(slide_index)]["acq" + \
                                                                                                    str(acq_index)][
                                    identifier] = convert_to_below_255(image)
                                if image_index == 1:
                                    description = f"{'acq' + str(acq_index)}, Dimensions: {image.shape[1]}x" \
                                                  f"{image.shape[0]}, Panel: {len(acq_text_read.channel_names)} markers"
                                    dataset_information.append(description)
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
    for exp in uploaded.keys():
        if "metadata" not in exp:
            current_blend_dict[exp] = {}
            for slide in uploaded[exp].keys():
                current_blend_dict[exp][slide] = {}
                for acq in uploaded[exp][slide].keys():
                    current_blend_dict[exp][slide][acq] = {}
                    for channel in uploaded[exp][slide][acq].keys():
                        current_blend_dict[exp][slide][acq][channel] = {'color': None,
                                                                        'x_lower_bound': None,
                                                                        'x_upper_bound': None,
                                                                        'y_ceiling': None,
                                                                        'filter_type': None,
                                                                        'filter_val': None}
                        current_blend_dict[exp][slide][acq][channel]['color'] = '#FFFFFF'
    return current_blend_dict


def populate_upload_dict_by_roi(upload_dict, dataset_selection, session_config):
    """
    Populate an existing upload dictionary with an ROI read from a filepath for lazy loading
    """
    try:
        split = dataset_selection.split("+")
        exp, slide, acq_name = split[0], split[1], split[2]
        # get the index of the file from the experiment number in the event that there are multiple uploads
        index = int(exp.split("experiment")[1])
        file_path = session_config['uploads'][index]
        assert file_path.endswith('.mcd')
        with MCDFile(file_path) as mcd_file:
            for slide_inside in mcd_file.slides:
                for acq in slide_inside.acquisitions:
                    if acq.description == acq_name:
                        channel_names = acq.channel_names
                        channel_index = 0
                        img = mcd_file.read_acquisition(acq)
                        for channel in img:
                            upload_dict[exp][slide][acq_name][channel_names[channel_index]] = channel
                            channel_index += 1
        return upload_dict
    except (KeyError, AssertionError):
        return upload_dict
