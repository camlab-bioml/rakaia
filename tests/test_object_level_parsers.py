import pytest
from dash_uploader import UploadStatus
import dash_extensions
import pandas as pd
import dash
import os
import numpy as np
import re
from dash.exceptions import PreventUpdate
from rakaia.parsers.object import (
    validate_incoming_measurements_csv,
    validate_imported_csv_annotations,
    QuantificationFormatError,
    filter_measurements_csv_by_channel_percentile,
    get_quantification_filepaths_from_drag_and_drop,
    parse_and_validate_measurements_csv,
    parse_masks_from_filenames,
    read_in_mask_array_from_filepath,
    set_columns_to_drop,
    set_mandatory_columns,
    RestyleDataParser,
    parse_roi_query_indices_from_quantification_subset,
    match_steinbock_mask_name_to_mcd_roi,
    match_mask_name_to_quantification_sheet_roi,
    ROIMaskMatch,
    validate_coordinate_set_for_image,
    parse_quantification_sheet_from_h5ad,
    validate_quantification_from_anndata,
    umap_dataframe_from_quantification_dict,
    GatingObjectList,
    is_steinbock_intensity_anndata,
    quant_dataframe_to_anndata,
    umap_transform, apply_gating_to_all_rois)
import anndata as adata

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

def test_validate_incoming_point_annotations(get_current_dir):
    points = pd.read_csv(os.path.join(get_current_dir, 'point_annotations.csv'))
    assert validate_imported_csv_annotations(points)
    points.columns = ["ROI", "centroid-0", "centroid-1", "annot", "col"]
    assert not validate_imported_csv_annotations(points)

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
    uploader = UploadStatus(uploaded_files=["measurements.csv"], n_total=1, uploaded_size_mb=0, total_size_mb=1)
    assert isinstance(get_quantification_filepaths_from_drag_and_drop(uploader), dash._callback.NoUpdate)


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
    uploader = UploadStatus(uploaded_files=["mask.tiff"], n_total=1, uploaded_size_mb=0, total_size_mb=1)
    assert isinstance(parse_masks_from_filenames(uploader), dash._callback.NoUpdate)

def test_read_in_mask_from_filepath(get_current_dir):
    masks_dict = {"mask": os.path.join(get_current_dir, "mask.tiff")}
    mask_return = read_in_mask_array_from_filepath(masks_dict, "mask", 1, None, True)
    assert isinstance(mask_return[0], dash_extensions.enrich.Serverside)
    assert isinstance(mask_return[1], list)
    assert 'mask' in mask_return[1]

    masks_dict_2 = {"mask_1": os.path.join(get_current_dir, "mask.tiff"),
                    "mask_2": os.path.join(get_current_dir, "mask.tiff")}
    mask_return = read_in_mask_array_from_filepath(masks_dict_2, "mask", 1, None, False)
    assert isinstance(mask_return[0], dash_extensions.enrich.Serverside)
    assert isinstance(mask_return[1], list)
    assert 'mask_2' in mask_return[1]

def test_return_proper_cols_remove_validate():
    assert 'cell_id' in set_columns_to_drop()
    assert 'x_min' in set_columns_to_drop()
    assert 'x_min' not in set_mandatory_columns()
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

    types_return_1 = RestyleDataParser(restyle_1, test_frame, "category", None).get_callback_structures()
    assert types_return_1 == (['two'], [1])

    restyle_2 = [{'visible': [True]}, [6]]
    types_return_2 = RestyleDataParser(restyle_2, test_frame, "category", [1]).get_callback_structures()
    assert types_return_2 == (['two', 'seven'], [1, 6])

    restyle_3 = [{'visible': ['legendonly']}, [3]]
    types_return_3 = RestyleDataParser(restyle_3, test_frame, "category", [0, 1, 2, 3]).get_callback_structures()
    assert types_return_3 == (['one', 'two', 'three'], [0, 1, 2])
    assert RestyleDataParser([{'visible': ['legendonly']}, [0]],
        test_frame, "category", [0, 1, 2, 3]).get_callback_structures() == (None, None)

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
    assert ROIMaskMatch(data_selection, mask_options, None).get_match() == "roi_1"
    assert ROIMaskMatch(data_selection, mask_options, None, return_as_dash=True).get_match() == "roi_1"

    data_selection_2 = "roi_1+++slide0+++roi_1"
    assert ROIMaskMatch(data_selection_2, mask_options, None).get_match() == "roi_1"

    data_selection = "MCD1+++slide0+++roi_1"
    mask_options = ["roi_1_mask", "roi_2_mask"]
    assert ROIMaskMatch(data_selection, mask_options, None).get_match() == "roi_1_mask"

    dataset_options = ["round_1", "round_2", "round_3", "round_4"]
    mask_options = ["mcd1_s0_a1_ac_IA_mask", "mcd1_s0_a_ac_IA_mask", "mcd1_s0_a3_ac_IA_mask", "mcd1_s0_a4_ac_IA_mask"]

    assert ROIMaskMatch("round_3", mask_options, dataset_options).get_match() == "mcd1_s0_a3_ac_IA_mask"
    # Expect an index error on the second mask name as it's malformed
    assert not ROIMaskMatch("round_2", mask_options, dataset_options).get_match()

    dataset_options = ["round_1", "round_2", "MCD1+++slide0+++roi_1", "round_4"]
    mask_options = ["mcd1_s0_a1_ac_IA_mask", "mcd1_s0_a2_ac_IA_mask", "mcd1_s0_a3_ac_IA_mask", "mcd1_s0_a4_ac_IA_mask"]

    assert ROIMaskMatch("MCD1+++slide0+++roi_1", mask_options, dataset_options).get_match() == "mcd1_s0_a3_ac_IA_mask"

    assert ROIMaskMatch("MCD1+++slide0+++roi_1", None, dataset_options).get_match() is None

    assert ROIMaskMatch("MCD1+++slide0+++roi_1", [], dataset_options).get_match() is None
    assert isinstance(ROIMaskMatch("MCD1+++slide0+++roi_1", [], dataset_options, return_as_dash=True).get_match(),
                      dash._callback.NoUpdate)

    # with steinbock-style mask naming
    data_selection = "Patient1+++slide0+++pos_1_3_3"
    mask_options = ["Patient1_002", "Patient1_003"]
    assert ROIMaskMatch(data_selection, mask_options, None).get_match() == "Patient1_003"

    data_selection = "patient7B_SPS23_836_2_3---slide0---ROI_006_6"
    mask_options = ["patient7B_SPS23_836_2_3_002", "patient7B_SPS23_836_2_3_006", "patient7B_SPS23_836_2_3_007"]
    assert ROIMaskMatch(data_selection, mask_options, None, delimiter="---").get_match() == "patient7B_SPS23_836_2_3_006"


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

    mask_name = "patient1_003"
    cell_id_list = ["pos1_3_3", "Other"]
    assert match_mask_name_to_quantification_sheet_roi(mask_name, cell_id_list) == "pos1_3_3"

    mask_name = "patient7B_SPS23_836_2_3_11"
    cell_id_list = ["ROI_011_11", "ROI_012_12", "Other_mask"]
    assert match_mask_name_to_quantification_sheet_roi(mask_name, cell_id_list) == "ROI_011_11"

    mask_name = "patient1_002"
    cell_id_list = ["pos1_3_3", "Other"]
    assert not match_mask_name_to_quantification_sheet_roi(mask_name, cell_id_list)


def test_validate_xy_coordinates_for_image():
    image = np.full((1000, 100, 3), 255)
    assert validate_coordinate_set_for_image(x_coord=10, y_coord=10, image=image)
    assert not validate_coordinate_set_for_image(x_coord=101, y_coord=10, image=image)
    assert not validate_coordinate_set_for_image()

def test_parse_quantification_sheet_from_anndata(get_current_dir):
    anndata = os.path.join(get_current_dir, "quantification_anndata.h5ad")
    assert not is_steinbock_intensity_anndata(adata.read_h5ad(anndata))
    quant_sheet = parse_quantification_sheet_from_h5ad(anndata)
    assert quant_sheet.shape == (1445, 24)
    assert 'cell_id' in quant_sheet.columns
    assert 'sample' in quant_sheet.columns

    # check that the correct columns get dropped before UMAP computation
    cols_drop = set_columns_to_drop(quant_sheet)
    assert cols_drop == ['cell_id', 'sample', 'x', 'y', 'size', 'leiden', 'cluster_id']

    anndata_frame, placeholder = validate_quantification_from_anndata(anndata)
    assert anndata_frame.shape == (1445, 7)

def test_parse_quantification_sheet_from_anndata_steinbock(get_current_dir):
    anndata = os.path.join(get_current_dir, "from_steinbock.h5ad")
    assert is_steinbock_intensity_anndata(adata.read_h5ad(anndata))
    quant_sheet = parse_quantification_sheet_from_h5ad(anndata)
    assert all([re.search(r'\d+$', elem).group() for elem in quant_sheet['sample'].to_list()])
    assert all([col in quant_sheet.columns for col in ['sample', 'cell_id']])
    assert quant_sheet.shape[0] == 669

def test_return_umap_dataframe_from_quantification_dict(get_current_dir):
    quant_sheet = pd.DataFrame({'Channel_1': [1, 2, 3, 4, 5, 6], 'Channel_2': [1, 2, 3, 4, 5, 6]})
    cur_umap = pd.DataFrame({'UMAP1': [1, 2, 3, 4, 5, 6], 'UMAP2': [1, 2, 3, 4, 5, 6]})
    umap = umap_dataframe_from_quantification_dict(quant_sheet, cur_umap, rerun=False,
                                                   cols_include=['Channel_1'])
    assert isinstance(umap, dash._callback.NoUpdate)

def test_vary_umap_transform(get_current_dir):
    anndata = adata.read_h5ad(os.path.join(get_current_dir, "quantification_anndata.h5ad"))
    expr = pd.DataFrame(anndata.X, columns= anndata.var_names)
    pca = umap_transform(expr, use_pca=True)
    scaled = umap_transform(expr)
    assert scaled.shape == expr.shape
    assert pca.shape != scaled.shape

    umap = umap_dataframe_from_quantification_dict(expr.sample(n = 100), min_dist=1)
    assert umap.value.shape == (100, 2)

def test_csv_to_anndata(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    adata_obj = quant_dataframe_to_anndata(measurements_csv)
    assert list(adata_obj.obs.columns)[0] == "sample"
    assert len(adata_obj.var_names) == adata_obj.X.shape[1]
    measurements_back = parse_quantification_sheet_from_h5ad(adata_obj)
    assert measurements_csv.equals(measurements_back)

def test_gating_cell_ids(get_current_dir):

    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    gating_selection = ['191Ir_DNA1', '168Er_Ki67']
    gating_dict = {'191Ir_DNA1': {'lower_bound': 0.2, 'upper_bound': 0.4},
                   '168Er_Ki67': {'lower_bound': 0.5, 'upper_bound': 1}}
    gating_objects = GatingObjectList(gating_dict, gating_selection, measurements_csv, "test_1_mask")
    cell_ids = gating_objects.get_object_list()
    # test 1 has only cells up to 203, so can enforce that only one ROI was used
    assert max(cell_ids) < 203
    assert len(cell_ids) > 0

    gating_objects_2 = GatingObjectList(gating_dict, gating_selection, measurements_csv, "test_2_mask")
    cell_ids_2 = gating_objects_2.get_object_list()
    all_indices = gating_objects.get_query_indices_all()
    assert len(all_indices) == (len(cell_ids) + len(cell_ids_2))

    # apply to all
    apply_all = apply_gating_to_all_rois(measurements_csv, all_indices, 'gating_test',
                                         'threshold_1', as_dict=False)
    breakdown = apply_all['gating_test'].value_counts().to_dict()
    assert int(breakdown['threshold_1']) == len(all_indices)

    # Apply a second category to all, then remove both gate annotations

    gating_dict_2 = {'191Ir_DNA1': {'lower_bound': 0.25, 'upper_bound': 0.4},
                   '168Er_Ki67': {'lower_bound': 0.7, 'upper_bound': 1}}
    gating_objects_2 = GatingObjectList(gating_dict_2, gating_selection, measurements_csv, "test_1_mask")
    cell_ids_2 = gating_objects_2.get_object_list()
    # test 1 has only cells up to 203, so can enforce that only one ROI was used
    assert len(cell_ids_2) < len(cell_ids)
    assert len(cell_ids_2) > 0
    all_indices_2 = gating_objects_2.get_query_indices_all()

    apply_all_2 = apply_gating_to_all_rois(apply_all, all_indices_2, 'gating_test',
                                         'threshold_2', as_dict=False)

    assert len(apply_all_2['gating_test'].value_counts()) == 3
    assert 'threshold_2' in list(apply_all_2['gating_test'].value_counts().to_dict().keys())

    remove_last = apply_gating_to_all_rois(apply_all_2, all_indices_2, 'gating_test',
                                'threshold_2', reset_to_default=True, as_dict=False)
    assert len(remove_last['gating_test'].value_counts()) == 2
    assert 'threshold_2' not in list(remove_last['gating_test'].value_counts().to_dict().keys())

    remove_all = apply_gating_to_all_rois(remove_last, all_indices, 'gating_test',
                                           'threshold_1', reset_to_default=True, as_dict=False)
    assert len(remove_all['gating_test'].value_counts()) == 1
    assert list(remove_all['gating_test'].value_counts().to_dict().keys()) == ['Unassigned']

    no_change = apply_gating_to_all_rois(measurements_csv, all_indices, None, None)
    assert 'gating_test' not in list(pd.DataFrame(no_change).columns)

    cell_id_intersection = GatingObjectList(gating_dict, gating_selection, measurements_csv, "test_1",
                                                      intersection=True).get_object_list()
    assert len(cell_ids) > len(cell_id_intersection)

    fake_frame = pd.DataFrame({"191Ir_DNA1": [1, 2, 3, 4, 5],
                               "168Er_Ki67": [1, 2, 3, 4, 5]})

    # if there is no column to match the ids to the mask, return empty
    assert GatingObjectList(gating_dict, gating_selection, fake_frame, "test_1",
                                                      intersection=True).get_object_list() == []

    fake_frame = pd.DataFrame({"191Ir_DNA1": [1, 2, 3, 4, 5],
                               "168Er_Ki67": [1, 2, 3, 4, 5],
                               "description": ["roi", "roi", "roi", "roi", "roi"]})
    assert GatingObjectList(gating_dict, gating_selection, fake_frame, "test_1",
                                      intersection=True).get_object_list() == []

def test_match_steinbock_mask_name_to_roi():
    assert match_steinbock_mask_name_to_mcd_roi("patient1_003", "pos_1_3_3") == "patient1_003"
    assert match_steinbock_mask_name_to_mcd_roi("patient1_003", "pos_1_3_3", False) == "pos_1_3_3"
    assert not match_steinbock_mask_name_to_mcd_roi("patient1_003", "pos_1_3_2")
    assert match_steinbock_mask_name_to_mcd_roi("pos1_3_003", "pos_1_3_3") == "pos1_3_003"

    assert not match_steinbock_mask_name_to_mcd_roi("file_1_name", "pos_1_3_3")
    assert not match_steinbock_mask_name_to_mcd_roi("patient1_003", "3_this_is_an_roi")
