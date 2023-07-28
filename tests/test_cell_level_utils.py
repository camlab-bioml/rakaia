import pytest

from ccramic.app.utils.cell_level_utils import *
from ccramic.app.parsers.cell_level_parsers import *
from ccramic.app.inputs.cell_level_inputs import *
import os
from PIL import Image
import dash_extensions

def test_basic_mask_boundary_converter(get_current_dir):
    fake_mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    mask_copy = fake_mask.copy()
    assert np.array_equal(fake_mask, mask_copy)
    mask_with_boundary = convert_mask_to_cell_boundary(fake_mask)
    assert not np.array_equal(fake_mask, mask_with_boundary)
    assert np.max(mask_with_boundary) == 255
    assert not np.max(fake_mask) == 255
    reconverted_back = np.array(Image.fromarray(fake_mask).convert('RGB').convert('L'))
    assert np.max(reconverted_back) == 255
    # assert that the boundary conversion makes the overall mean less because white pixels on the interior of
    # the cell are converted to black
    # need to convert the mask to RGB then back to greyscale to ensure that the mask max is 255 for standard intensity
    # comparison
    mask_from_reconverted = convert_mask_to_cell_boundary(reconverted_back)
    assert np.mean(reconverted_back) > np.mean(mask_from_reconverted)
    # assert that a pixel inside the cell boundary is essentially invisible
    assert reconverted_back[628, 491] == 255
    assert mask_from_reconverted[628, 491] <= 3

def test_umap_from_quantification_dict(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements = parse_and_validate_measurements_csv(measurements_dict)
    returned_umap = return_umap_dataframe_from_quantification_dict(validated_measurements)
    assert isinstance(returned_umap, tuple)
    assert isinstance(returned_umap[0], dash_extensions.enrich.Serverside)
    assert isinstance(returned_umap[1], list)
    with pytest.raises(PreventUpdate):
        return_umap_dataframe_from_quantification_dict(None)

def test_receive_alert_on_imcompatible_mask():
    upload_dict = {"experiment0": {"slide0": {"acq0": {"channel_1": np.empty((50, 50))}}}}
    data_selection = "experiment0+++slide0+++acq0"
    mask_dict = {"mask": np.empty((50, 49))}
    error = send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
    assert 'error' in error.keys()
    with pytest.raises(PreventUpdate):
        # if the masks are the same, do not send error
        upload_dict = {"experiment0": {"slide0": {"acq0": {"channel_1": np.empty((50, 50))}}}}
        data_selection = "experiment0+++slide0+++acq0"
        mask_dict = {"mask": np.empty((50, 50))}
        send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
    with pytest.raises(PreventUpdate):
        # if inputs are none, do not set error
        send_alert_on_incompatible_mask(None, None, None, None, "mask", True)
