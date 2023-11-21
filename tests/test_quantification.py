import numpy as np
from PIL import Image
import os
from ccramic.utils.quantification import (
    quantify_one_channel,
    quantify_multiple_channels_per_roi,
    concat_quantification_frames_multi_roi)

def test_quantification_one_channel(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    array = np.full((mask.shape[0], mask.shape[1]), 7)
    mean_values = quantify_one_channel(array, mask)
    # assert that the number of entries in the vector is equal to the number of cells in the mask
    assert mean_values.shape[1] == int(np.max(mask))

    bad_array = np.full((600, mask.shape[1]), 7)
    assert quantify_one_channel(bad_array, mask) is None

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
    assert column_list.index('sample') == 3
    assert all([elem in column_list[0:3] for elem in channel_list])

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
