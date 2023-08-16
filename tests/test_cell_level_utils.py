import numpy as np
import pytest

from ccramic.app.utils.cell_level_utils import *
from ccramic.app.parsers.cell_level_parsers import *
from ccramic.app.inputs.cell_level_inputs import *
import os
from PIL import Image
import dash_extensions

def test_basic_col_dropper(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    assert "x_max" in measurements.columns
    measurements = drop_columns_from_measurements_csv(measurements)
    for col in set_columns_to_drop():
        assert col not in measurements.columns

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
    validated_measurements, cols = parse_and_validate_measurements_csv(measurements_dict)
    returned_umap = return_umap_dataframe_from_quantification_dict(validated_measurements)
    assert isinstance(returned_umap, tuple)
    assert isinstance(returned_umap[0], dash_extensions.enrich.Serverside)
    assert isinstance(returned_umap[1], list)
    with pytest.raises(PreventUpdate):
        return_umap_dataframe_from_quantification_dict(None)

def test_receive_alert_on_imcompatible_mask():
    upload_dict = {"experiment0": {"slide0": {"acq0": {"channel_1": np.empty((50, 50))}}}}
    data_selection = "experiment0+++slide0+++acq0"
    mask_dict = {"mask": {"array": np.empty((50, 49))}}
    error = send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
    assert 'error' in error.keys()
    with pytest.raises(PreventUpdate):
        # if the masks are the same, do not send error
        upload_dict = {"experiment0": {"slide0": {"acq0": {"channel_1": np.empty((50, 50))}}}}
        data_selection = "experiment0+++slide0+++acq0"
        mask_dict = {"mask": {"array": np.empty((50, 50))}}
        send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
    with pytest.raises(PreventUpdate):
        # if inputs are none, do not set error
        send_alert_on_incompatible_mask(None, None, None, None, "mask", True)


def test_basic_parser_bounding_box_min_max():
    bounds = {'xaxis.range[0]': 241, 'xaxis.range[1]': 253, 'yaxis.range[0]': -1, 'yaxis.range[1]': 4}
    x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(bounds)
    assert x_min == 241
    assert x_max == 253
    assert y_min == -1
    assert y_max == 4
    bounds_2 = {'xaxis.range[1]': 241, 'xaxis.range[0]': 253, 'yaxis.range[1]': -1, 'yaxis.range[0]': 4}
    x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(bounds_2)
    assert x_min == 241
    assert x_max == 253
    assert y_min == -1
    assert y_max == 4
    bounds_bad_name = {'fake': 241, 'xaxis.range[0]': 253, 'yaxis.range[1]': -1, 'yaxis.range[0]': 4}
    assert get_min_max_values_from_zoom_box(bounds_bad_name) is None

def test_basic_box_query_measurements_csv(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    coord_dict = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}
    assert len(subset_measurements_by_cell_graph_box(measurements, coord_dict)) == 1
    measurements_2 = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    assert subset_measurements_by_cell_graph_box(measurements_2, coord_dict) is None

def test_basic_cell_annotation_col_pop(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    bounds = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}

    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds,
                                                                   cell_type="new_cell_type")
    assert list(measurements["ccramic_cell_annotation"][(measurements["x_max"] == 836) &
          (measurements["y_max"] == 20)]) == ['new_cell_type']
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 1
    counts = measurements["ccramic_cell_annotation"].value_counts(normalize =True)
    assert len(dict(counts)) == 2
    assert 'None' in dict(counts).keys()

    bounds_2 = {'xaxis.range[0]': 241, 'xaxis.range[1]': 253, 'yaxis.range[0]': -1, 'yaxis.range[1]': 4}

    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds_2,
                                                                     cell_type="new_cell_type")
    assert list(measurements["ccramic_cell_annotation"][(measurements["x_max"] == 836) &
                                                        (measurements["y_max"] == 20)]) == ['new_cell_type']
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 2
    assert len(dict(counts)) == 2
    assert 'None' in dict(counts).keys()
    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds_2,
                                                                    cell_type="new_cell_type_2")
    counts = measurements["ccramic_cell_annotation"].value_counts(normalize=True)
    assert len(dict(counts)) == 3
    assert 'new_cell_type_2' in dict(counts).keys()


def test_convert_basic_array_to_hovertemplate():
    array = np.zeros((1000, 1000))
    assert len(array.shape) == 2
    template = process_mask_array_for_hovertemplate(array)
    assert template.shape[2] == 1
    assert np.unique(template) == ['None']
