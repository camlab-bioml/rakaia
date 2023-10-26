import pytest

from ccramic.app.parsers.cell_level_parsers import *
from ccramic.app.utils.cell_level_utils import *
import numpy as np
import pandas as pd
import os
from dash_uploader import UploadStatus
from dash.exceptions import PreventUpdate
import dash_extensions

def test_validation_of_measurements_csv(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    valid, err = validate_incoming_measurements_csv(measurements_csv)
    assert measurements_csv.equals(valid)
    assert valid is not None
    assert err is None

    # currently, validation requires only sample to pass
    measurements_bad = measurements_csv.drop(['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample'], axis=1)
    valid_bad, err = validate_incoming_measurements_csv(measurements_bad)
    assert valid_bad is None
    assert err is None

    fake_image = np.empty((1490, 93, 3))
    valid, err = validate_incoming_measurements_csv(measurements_csv, current_image=fake_image)
    assert valid is not None
    assert err is None

    # fake_image_bad_dims = np.empty((1490, 92, 3))
    # not_valid, err = validate_incoming_measurements_csv(measurements_csv, current_image=fake_image_bad_dims)
    # assert not_valid is not None
    # assert err is not None


def test_filtering_channel_measurements_by_percentile(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    filtered = filter_measurements_csv_by_channel_percentile(measurements_csv, drop_cols=True)
    assert len(measurements_csv) > len(filtered)
    for col in filtered.columns:
        assert np.max(measurements_csv[col]) > np.max(filtered[col])

    filtered_50 = filter_measurements_csv_by_channel_percentile(measurements_csv, percentile=0.5, drop_cols=True)
    for col in filtered_50.columns:
        assert np.max(filtered[col]) > np.max(filtered_50[col])


def test_parsing_quantification_filepaths():
    uploader = UploadStatus(uploaded_files=["measurements.csv"], n_total=1, uploaded_size_mb=1, total_size_mb=1)
    upload_session = get_quantification_filepaths_from_drag_and_drop(uploader)
    assert len(upload_session['uploads']) > 0

def test_parsing_incoming_measurements_csv(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols, err = parse_and_validate_measurements_csv(measurements_dict)
    assert isinstance(validated_measurements, list)
    for elem in validated_measurements:
        assert isinstance(elem, dict)
    with pytest.raises(PreventUpdate):
        parse_and_validate_measurements_csv(None)
    with pytest.raises(PreventUpdate):
        measurements_dict = {"fake_col": [os.path.join(get_current_dir, "cell_measurements.csv")]}
        parse_and_validate_measurements_csv(measurements_dict)

def test_parse_mask_filenames():
    uploader = UploadStatus(uploaded_files=["mask.tiff"], n_total=1, uploaded_size_mb=1, total_size_mb=1)
    mask_files = parse_masks_from_filenames(uploader)
    assert 'mask' in mask_files.keys()
    assert mask_files['mask'] == "mask.tiff"

def test_read_in_mask_from_filepath(get_current_dir):
    masks_dict = {"mask": os.path.join(get_current_dir, "mask.tiff")}
    #TODO: validate the read_in_mask_array_from_filepath function
    mask_return = read_in_mask_array_from_filepath(masks_dict, "mask", 1, None, True)
    assert isinstance(mask_return[0], dash_extensions.enrich.Serverside)
    assert isinstance(mask_return[1], list)
    assert 'mask' in mask_return[1]

    masks_dict_2 = {"mask_1": os.path.join(get_current_dir, "mask.tiff"),
                    "mask_2": os.path.join(get_current_dir, "mask.tiff")}
    # TODO: validate the read_in_mask_array_from_filepath function
    mask_return = read_in_mask_array_from_filepath(masks_dict_2, "mask", 1, None, True)
    assert isinstance(mask_return[0], dash_extensions.enrich.Serverside)
    assert isinstance(mask_return[1], list)
    assert 'mask_2' in mask_return[1]

def test_return_proper_cols_remove_validate():
    assert 'cell_id' in set_columns_to_drop()
    assert 'x_min' in set_columns_to_drop()
    assert not 'x_min' in set_mandatory_columns()
    assert len(set_columns_to_drop()) != len(set_mandatory_columns())

def test_parse_restyledata_from_legend_change():
    """
    test that the dictionary from a legend restyle change in the dcc.Graph can be registered properly
    """
    test_frame = {"category": ["one", "two", "three", "four", "five", "six", "seven"],
                 "value": [1, 2, 3, 4, 5, 6, 7]}
    # case 1: when all categories are selected
    restyle_1 = [{'visible': ['legendonly', True, 'legendonly', 'legendonly', 'legendonly', 'legendonly', 'legendonly']},
                 [0, 1, 2, 3, 4, 5, 6]]

    types_return_1 = parse_cell_subtypes_from_restyledata(restyle_1, test_frame, "category", None)
    assert types_return_1 == (['two'], [1])

    restyle_2 = [{'visible': [True]}, [6]]
    types_return_2 = parse_cell_subtypes_from_restyledata(restyle_2, test_frame, "category", [1])
    assert types_return_2 == (['two', 'seven'], [1, 6])

    restyle_3 = [{'visible': ['legendonly']}, [3]]
    types_return_3 = parse_cell_subtypes_from_restyledata(restyle_3, test_frame, "category", [0, 1, 2, 3])
    assert types_return_3 == (['one', 'two', 'three'], [0, 1, 2])
