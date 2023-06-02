import pandas as pd
import h5py
from pathlib import Path
from tifffile import TiffFile
import os
from .utils import *
from readimc import MCDFile


def populate_upload_dict(uploaded_files):
    """
    Populate a nested dictionary based on the uploaded files.
    """
    filenames = [str(x) for x in uploaded_files]
    upload_dict = {}
    unique_image_names = []
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
                            for channel in data_h5[exp][slide][acq]:
                                upload_dict[exp][slide][acq][channel] = data_h5[exp][slide][acq][channel]['image'][()]
                                if channel not in unique_image_names:
                                    unique_image_names.append(channel)
                                blend_dict[exp][slide][acq][channel] = {}
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
                                assert all(len(tif.pages) == len(value) for value in \
                                           list(upload_dict['metadata'].values()))

                            file__name, file_extension = os.path.splitext(tiff_path)
                            # set different image labels based on the basename of the file (ome.tiff vs .tiff)
                            # if "ome" in upload:
                            #     basename = str(os.path.basename(tiff_path)).split(".ome" + file_extension)[0]
                            # else:
                            #     basename = str(os.path.basename(tiff_path)).split(file_extension)[0]
                            multi_channel_index = 0
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
                                multi_channel_index += 1
                                metadata_channels.append(identifier)
                                metadata_labels.append(identifier)
                                if identifier not in unique_image_names:
                                    unique_image_names.append(identifier)

                            if len(upload_dict['metadata']) < 1:
                                upload_dict['metadata'] = {'Cycle': range(1, len(metadata_channels) + 1, 1),
                                               'Channel Name': metadata_channels,
                                               'Channel Label': metadata_labels,
                                               'ccramic Label': metadata_labels}
                                upload_dict['metadata_columns'] = ['Cycle', 'Channel Name', 'Channel Label', 'ccramic Label']
                        acq_index += 1
                    except AssertionError:
                        pass
                elif upload.endswith('.mcd'):
                    upload_dict["experiment" + str(experiment_index)] = {}
                    with MCDFile(upload) as mcd_file:
                        channel_names = None
                        channel_labels = None
                        slide_index = 0
                        acq_index = 0
                        for slide in mcd_file.slides:
                            upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)] = {}
                            acq_index = 0
                            for acq in slide.acquisitions:
                                upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)][
                                    "acq" + str(acq_index)] = {}
                                if channel_labels is None:
                                    channel_labels = acq.channel_labels
                                    channel_names = acq.channel_names
                                    upload_dict['metadata'] = {'Cycle': range(1, len(channel_names) + 1, 1),
                                                               'Channel Name': channel_names,
                                                               'Channel Label': channel_labels,
                                                               'ccramic Label': channel_labels}
                                    upload_dict['metadata_columns'] = ['Cycle', 'Channel Name', 'Channel Label',
                                                                       'ccramic Label']
                                else:
                                    assert all(label in acq.channel_labels for label in channel_labels)
                                    assert all(name in acq.channel_names for name in channel_names)
                                img = mcd_file.read_acquisition(acq)
                                channel_index = 0
                                for channel in img:
                                    upload_dict["experiment" + str(experiment_index)]["slide" +
                                                                                      str(slide_index)]["acq" +
                                                                                                        str(acq_index)][
                                        channel_names[channel_index]] = convert_to_below_255(channel)
                                    if channel_names[channel_index] not in unique_image_names:
                                        unique_image_names.append(channel_names[channel_index])
                                    channel_index += 1
                                acq_index += 1
                            slide_index += 1
                    experiment_index += 1
        return upload_dict, blend_dict, unique_image_names
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
