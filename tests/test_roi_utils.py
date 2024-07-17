import pandas as pd
import os
import numpy as np
from PIL import Image
from rakaia.utils.object import convert_mask_to_object_boundary
from rakaia.utils.roi import (
    generate_dict_of_roi_cell_ids,
    subset_mask_outline_using_cell_id_list,
    override_roi_gallery_blend_list)
from rakaia.utils.region import check_for_valid_annotation_hash

def test_generate_dict_of_roi_cell_ids(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    cell_id_dict = generate_dict_of_roi_cell_ids(measurements)
    assert len(cell_id_dict) == 1
    assert 'Dilution_series_1_1' in cell_id_dict.keys()
    assert len(cell_id_dict['Dilution_series_1_1']) == len(measurements)

    assert generate_dict_of_roi_cell_ids(measurements, "fake_col", "fake_col") is None

def test_basic_mask_boundary_subsetting(get_current_dir):
    """
    Test that the subsetting of the mask outline using the original mask works as intended
    """
    fake_mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    mask_with_boundary = convert_mask_to_object_boundary(fake_mask)
    assert np.max(mask_with_boundary) == 255
    outline_subset = subset_mask_outline_using_cell_id_list(fake_mask, mask_with_boundary, range(50, 100))
    assert np.max(mask_with_boundary) > np.max(outline_subset)

    assert subset_mask_outline_using_cell_id_list(np.zeros((90, 90)), mask_with_boundary, []) is None

def test_basic_check_annotation_hash():
    assert check_for_valid_annotation_hash() == {}
    assert check_for_valid_annotation_hash(None, "roi_1") == {"roi_1": {}}
    assert check_for_valid_annotation_hash({"roi_1": {}}, "roi_1") == {"roi_1": {}}

def test_override_roi_channel_list():
    assert override_roi_gallery_blend_list(["chan_1", "chan_2"], None, None) == ["chan_1", "chan_2"]
    assert override_roi_gallery_blend_list(["chan_1", "chan_2"], {"immune": ["CD3", "CD8"]},
            "immune") == ['CD3', 'CD8']
    assert override_roi_gallery_blend_list(["chan_1", "chan_2"], {"immune": ["CD3", "CD8"]},
                                           "infiltrate") == ["chan_1", "chan_2"]
