import dash
import pandas as pd
import pytest
from dash_uploader import UploadStatus
import dash_extensions
from ccramic.parsers.cell_level_parsers import *
from pandas.testing import assert_frame_equal

def test_validation_of_measurements_csv(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    valid, err = validate_incoming_measurements_csv(measurements_csv)
    assert measurements_csv.equals(valid)
    assert valid is not None
    assert err is None

    # currently, validation requires only sample to pass
    measurements_bad = measurements_csv.drop(['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample'], axis=1)
    with pytest.raises(QuantificationFormatError):
        validate_incoming_measurements_csv(measurements_bad)

    valid, err = validate_incoming_measurements_csv(measurements_csv)
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
    with pytest.raises(PreventUpdate):
        uploader = UploadStatus(uploaded_files=["measurements.csv"], n_total=1, uploaded_size_mb=0, total_size_mb=1)
        get_quantification_filepaths_from_drag_and_drop(uploader)


def test_parsing_incoming_measurements_csv(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols, err = parse_and_validate_measurements_csv(measurements_dict)
    assert isinstance(validated_measurements, list)
    for elem in validated_measurements:
        assert isinstance(elem, dict)

    measurements_dict = {"uploads": [os.path.join(get_current_dir, "quantification_anndata.h5ad")]}
    validated_measurements, cols, err = parse_and_validate_measurements_csv(measurements_dict)
    assert isinstance(validated_measurements, list)
    for elem in validated_measurements:
        assert isinstance(elem, dict)

    # assert that a fake sheet returns the appropriate updates

    measurements_dict = {"uploads": [os.path.join(get_current_dir, "this_file_is_fake.txt")]}
    validated_measurements, cols, err = parse_and_validate_measurements_csv(measurements_dict)
    assert validated_measurements is None
    assert err is not None

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
    with pytest.raises(PreventUpdate):
        uploader = UploadStatus(uploaded_files=["mask.tiff"], n_total=1, uploaded_size_mb=0, total_size_mb=1)
        parse_masks_from_filenames(uploader)

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
    mask_return = read_in_mask_array_from_filepath(masks_dict_2, "mask", 1, None, False)
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
    assert parse_cell_subtypes_from_restyledata([{'visible': ['legendonly']}, [0]],
                                                test_frame, "category", [0, 1, 2, 3]) == (None, None)

def test_valid_parse_for_indices_for_query(get_current_dir):
    """
    test that the parser for the measurements CSV is able to generate a list of valid indices
    """
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    expression = measurements_csv.drop(['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample'], axis=1)
    indices, counts = parse_roi_query_indices_from_quantification_subset(measurements_csv, expression, "sample")
    assert 'indices' in indices
    assert indices['indices'] == [0, 1]
    assert len(counts) == 2
    measurements_csv.rename(columns={"sample": "description"}, inplace=True)
    assert 'description' in measurements_csv.columns
    indices, counts = parse_roi_query_indices_from_quantification_subset(measurements_csv, expression, None)
    assert 'names' in indices
    assert indices['names'] == ['test_1', 'test_2']
    assert counts is None

def test_mask_match_to_roi_name():
    data_selection = "MCD1+++slide0+++roi_1"
    mask_options = ["roi_1", "roi_2"]
    assert match_mask_name_with_roi(data_selection, mask_options, None) == "roi_1"
    assert match_mask_name_with_roi(data_selection, mask_options, None, return_as_dash=True) == "roi_1"

    data_selection_2 = "roi_1+++slide0+++roi_1"
    assert match_mask_name_with_roi(data_selection_2, mask_options, None) == "roi_1"

    data_selection = "MCD1+++slide0+++roi_1"
    mask_options = ["roi_1_mask", "roi_2_mask"]
    assert match_mask_name_with_roi(data_selection, mask_options, None) == "roi_1_mask"

    dataset_options = ["round_1", "round_2", "round_3", "round_4"]
    mask_options = ["mcd1_s0_a1_ac_IA_mask", "mcd1_s0_a2_ac_IA_mask", "mcd1_s0_a3_ac_IA_mask", "mcd1_s0_a4_ac_IA_mask"]

    assert match_mask_name_with_roi("round_3", mask_options, dataset_options) == "mcd1_s0_a3_ac_IA_mask"

    dataset_options = ["round_1", "round_2", "MCD1+++slide0+++roi_1", "round_4"]
    mask_options = ["mcd1_s0_a1_ac_IA_mask", "mcd1_s0_a2_ac_IA_mask", "mcd1_s0_a3_ac_IA_mask", "mcd1_s0_a4_ac_IA_mask"]

    assert match_mask_name_with_roi("MCD1+++slide0+++roi_1", mask_options, dataset_options) == "mcd1_s0_a3_ac_IA_mask"

    assert match_mask_name_with_roi("MCD1+++slide0+++roi_1", None, dataset_options) is None

    assert match_mask_name_with_roi("MCD1+++slide0+++roi_1", [], dataset_options) is None
    assert isinstance(match_mask_name_with_roi("MCD1+++slide0+++roi_1", [], dataset_options, return_as_dash=True),
                      dash._callback.NoUpdate)

def test_match_mask_name_to_quantification_sheet_roi():
    samples = ["query_1", "query_2", "query_3", "query_4"]
    mask_selection = "query_s0_a2_ac_IA_mask"
    sam_match = match_mask_name_to_quantification_sheet_roi(mask_selection, samples)
    assert sam_match == "query_2"
    mask_selection_2 = "query_3"
    assert match_mask_name_to_quantification_sheet_roi(mask_selection_2, samples) == "query_3"
    assert match_mask_name_to_quantification_sheet_roi("query_5", samples) is None

    samples_no_index= ["sampletest"]
    assert match_mask_name_to_quantification_sheet_roi("sampletest", samples_no_index) == "sampletest"

    # assert partial match of the cell id ROI name to mask name works
    mask_name = "Kidney7_Sector2Row9Column6_SlideStart_mask"
    cell_id_list = ["Kidney7_Sector2Row9Column6_SlideStart", "Other"]
    assert match_mask_name_to_quantification_sheet_roi(mask_name, cell_id_list) == "Kidney7_Sector2Row9Column6_SlideStart"

    # assert that when the partial match doesn't work, it is None
    mask_name = "Kidney6_Sector2Row9Column6_SlideStart_mask"
    cell_id_list = ["Kidney7_Sector2Row9Column6_SlideStart", "Other"]
    assert match_mask_name_to_quantification_sheet_roi(mask_name, cell_id_list) is None

def test_validate_xy_coordinates_for_image():
    image = np.full((1000, 100, 3), 255)
    assert validate_coordinate_set_for_image(x=10, y=10, image=image)
    assert not validate_coordinate_set_for_image(x=101, y=10, image=image)
    assert not validate_coordinate_set_for_image()

def test_parse_quantification_sheet_from_anndata(get_current_dir):
    anndata = os.path.join(get_current_dir, "quantification_anndata.h5ad")
    quant_sheet = parse_quantification_sheet_from_h5ad(anndata)
    assert quant_sheet.shape == (1445, 24)
    assert 'cell_id' in quant_sheet.columns
    assert 'sample' in quant_sheet.columns

    # check that the correct columns get dropped before UMAP computation
    cols_drop = set_columns_to_drop(quant_sheet)
    assert cols_drop == ['cell_id', 'sample', 'x', 'y', 'size', 'leiden', 'cluster_id']

    anndata_frame, placeholder = validate_quantification_from_anndata(anndata)
    assert anndata_frame.shape == (1445, 7)

def test_return_umap_dataframe_from_quantification_dict(get_current_dir):
    quant_sheet = pd.DataFrame({'Channel_1': [1, 2, 3, 4, 5, 6], 'Channel_2': [1, 2, 3, 4, 5, 6]})
    cur_umap = pd.DataFrame({'UMAP1': [1, 2, 3, 4, 5, 6], 'UMAP2': [1, 2, 3, 4, 5, 6]})
    umap = return_umap_dataframe_from_quantification_dict(quant_sheet, cur_umap, rerun=False)
    assert isinstance(umap, dash._callback.NoUpdate)

def test_gating_cell_ids(get_current_dir):

    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    gating_selection = ['191Ir_DNA1', '168Er_Ki67']
    gating_dict = {'191Ir_DNA1': {'lower_bound': 0.2, 'upper_bound': 0.4},
                   '168Er_Ki67': {'lower_bound': 0.5, 'upper_bound': 1}}
    cell_ids = object_id_list_from_gating(gating_dict,gating_selection, measurements_csv, "test_1_mask")
    # test 1 has only cells up to 203, so can enforce that only one ROI was used
    assert max(cell_ids) < 203
    assert len(cell_ids) > 0
    cell_id_intersection = object_id_list_from_gating(gating_dict,gating_selection, measurements_csv, "test_1",
                                                      intersection=True)
    assert len(cell_ids) > len(cell_id_intersection)

    fake_frame = pd.DataFrame({"191Ir_DNA1": [1, 2, 3, 4, 5],
                               "168Er_Ki67": [1, 2, 3, 4, 5]})

    # if there is no column to match the ids to the mask, return empty
    assert object_id_list_from_gating(gating_dict, gating_selection, fake_frame, "test_1",
                                                      intersection=True) == []

    fake_frame = pd.DataFrame({"191Ir_DNA1": [1, 2, 3, 4, 5],
                               "168Er_Ki67": [1, 2, 3, 4, 5],
                               "description": ["roi", "roi", "roi", "roi", "roi"]})
    assert object_id_list_from_gating(gating_dict, gating_selection, fake_frame, "test_1",
                                      intersection=True) == []

def test_populating_cluster_annotation_dict():
    cluster_frame = pd.DataFrame({"cell_id": [1, 2, 3, 4, 5],
                                 "cluster": ["immune"] * 5})
    session_cluster_dict = cluster_annotation_frame_import(None, "roi_1", cluster_frame)
    assert_frame_equal(cluster_frame, session_cluster_dict['roi_1'])
    malformed = pd.DataFrame({"col_1": [1, 2, 3, 4, 5],
                                  "col_2": ["immune"] * 5})
    session_cluster_dict = cluster_annotation_frame_import(session_cluster_dict, "roi_2", malformed)
    assert "roi_2" not in session_cluster_dict.keys()
    assert "roi_1" in session_cluster_dict.keys()
