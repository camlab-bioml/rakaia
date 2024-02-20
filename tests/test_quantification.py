import numpy as np
from PIL import Image
import os
from ccramic.utils.quantification import (
    quantify_one_channel,
    quantify_multiple_channels_per_roi,
    concat_quantification_frames_multi_roi,
    populate_gating_dict_with_default_values,
    update_gating_dict_with_slider_values,
    gating_label_children)

def test_quantification_one_channel(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    array = np.full((mask.shape[0], mask.shape[1]), 7)
    mean_values = quantify_one_channel(array, mask)
    # assert that the number of entries in the vector is equal to the number of cells in the mask
    assert mean_values.shape[1] == int(np.max(mask))

    bad_array = np.full((600, mask.shape[1]), 7)
    assert quantify_one_channel(bad_array, mask) is None

def test_quantification_valid_vals(get_current_dir):
    """ Quantify one channel corresponding to Ki67
    Verified in the viewer that cell 797 has the highest mean expression"""
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    channel_dict = {"set1+++slide0+++roi_1": {"Ki67":
            np.array(Image.open(os.path.join(get_current_dir, "for_quant.tiff")))}}
    mean_values_ki67 = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_1",
                                                          ["Ki67"], mask_name="roi_1")
    where_max = mean_values_ki67.iloc[mean_values_ki67['Ki67'].idxmax()]
    assert where_max['cell_id'] == 797

    # assert where the largest cell is
    where_largest = mean_values_ki67.iloc[mean_values_ki67['area'].idxmax()]
    assert where_largest['cell_id'] == 1505

def test_quantification_multiple_channels(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    channel_dict = {"set1+++slide0+++roi_1": {"channel_1": np.full((mask.shape[0], mask.shape[1]), 1),
                 "channel_2": np.full((mask.shape[0], mask.shape[1]), 2),
                 "channel_3": np.full((mask.shape[0], mask.shape[1]), 3),
                 "channel_4": np.full((mask.shape[0], mask.shape[1]), 4)}}
    channel_list = ["channel_2", "channel_3", "channel_4"]
    multi_frame = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_1", channel_list)
    assert len(multi_frame.index) == int(np.max(mask))
    column_list = list(multi_frame.columns)
    assert 'sample' in multi_frame.columns
    assert column_list.index('sample') == 5
    assert all([elem in column_list[0:3] for elem in channel_list])

    dataset_options = ["set1+++slide0+++roi_1", "set1+++slide0+++roi_2"]

    aliases = {elem: elem for elem in channel_list}
    multi_frame = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_1", channel_list,
                                                     aliases, dataset_options)

    assert 'set1_1' in multi_frame['sample'].tolist()
    assert 'set1_1' not in multi_frame['description'].tolist()


def test_quantification_multiple_rois(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    channel_dict = {"set1+++slide0+++roi_1": {"channel_1": np.full((mask.shape[0], mask.shape[1]), 1),
                 "channel_2": np.full((mask.shape[0], mask.shape[1]), 2),
                 "channel_3": np.full((mask.shape[0], mask.shape[1]), 3),
                 "channel_4": np.full((mask.shape[0], mask.shape[1]), 4)},
                    "set1+++slide0+++roi_2": {"channel_1": np.full((mask.shape[0], mask.shape[1]), 5),
                     "channel_2": np.full((mask.shape[0], mask.shape[1]), 6),
                     "channel_3": np.full((mask.shape[0], mask.shape[1]), 7),
                     "channel_4": np.full((mask.shape[0], mask.shape[1]), 8)}}
    channel_list_1 = ["channel_2", "channel_3", "channel_4"]
    channel_list_2 = ["channel_1", "channel_2", "channel_3"]
    roi_1_quant = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_1", channel_list_1)
    roi_2_quant = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_2", channel_list_2)
    merged = concat_quantification_frames_multi_roi(roi_1_quant, roi_2_quant, "set1+++slide0+++roi_2")
    assert len(merged) == len(roi_1_quant)
    roi_2_quant = quantify_multiple_channels_per_roi(channel_dict, mask, "set1+++slide0+++roi_2", channel_list_1)
    merged = concat_quantification_frames_multi_roi(roi_1_quant, roi_2_quant, "set1+++slide0+++roi_2")
    assert len(merged) == 2 * len(roi_1_quant)

    merged = concat_quantification_frames_multi_roi(None, roi_2_quant, "set1+++slide0+++roi_2")
    assert len(merged) == len(roi_1_quant)
    assert 'roi_2' in merged['sample'].tolist()
    merged = concat_quantification_frames_multi_roi(roi_1_quant, None, "set1+++slide0+++roi_2")
    assert len(merged) == len(roi_1_quant)
    assert 'roi_1' in merged['sample'].tolist()


def test_populate_internal_gating_dict():
    channel_list = ["channel_1", "channel_2"]
    gating_dict = populate_gating_dict_with_default_values(None, channel_list)
    assert len(gating_dict) == 2
    gating_dict['channel_1'] = {'lower_bound': 0.2, 'upper_bound': 0.4}
    channel_list = ["channel_1", "channel_2", "channel_3"]
    gating_dict = populate_gating_dict_with_default_values(gating_dict, channel_list)
    assert len(gating_dict) == 3
    assert gating_dict['channel_1'] == {'lower_bound': 0.2, 'upper_bound': 0.4}

    modified_gating_dict = update_gating_dict_with_slider_values(gating_dict, "channel_3", [0.26, 0.28])
    assert modified_gating_dict['channel_3'] == {'lower_bound': 0.26, 'upper_bound': 0.28}

    modified_gating_dict = update_gating_dict_with_slider_values(modified_gating_dict,
                                                                 "channel_5", [0.33, 0.51])
    assert modified_gating_dict['channel_5'] == {'lower_bound': 0.33, 'upper_bound': 0.51}

def test_gating_label_children():
    gating_dict = {'channel_1': {'lower_bound': 0.2, 'upper_bound': 0.4}, 'channel_2': {'lower_bound': 0.0, 'upper_bound': 1.0},
     'channel_3': {'lower_bound': 0.26, 'upper_bound': 0.28}, 'channel_5': {'lower_bound': 0.33, 'upper_bound': 0.51}}
    current_gate = ["channel_1", "channel_2", "channel_6"]
    children = gating_label_children(True, gating_dict, current_gate)
    assert len(children) == 7
    assert not gating_label_children(False, gating_dict, current_gate)
