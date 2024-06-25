import collections
import pytest
import dash_extensions
from ccramic.utils.cell_level_utils import (
    send_alert_on_incompatible_mask,
    get_min_max_values_from_zoom_box,
    subset_measurements_by_cell_graph_box,
    populate_cell_annotation_column_from_bounding_box,
    process_mask_array_for_hovertemplate,
    get_cells_in_svg_boundary_by_mask_percentage,
    populate_cell_annotation_column_from_cell_id_list,
    populate_cell_annotation_column_from_clickpoint,
    subset_measurements_by_point,
    generate_greyscale_grid_array,
    identify_column_matching_roi_to_quantification,
    populate_quantification_frame_column_from_umap_subsetting,
    generate_mask_with_cluster_annotations,
    remove_annotation_entry_by_indices,
    quantification_distribution_table,
    custom_gating_id_list)
import pandas as pd
import os
import numpy as np
from PIL import Image
from ccramic.parsers.cell_level_parsers import (
    drop_columns_from_measurements_csv,
    set_columns_to_drop,
    convert_mask_to_cell_boundary,
    parse_quantification_sheet_from_h5ad,
    parse_and_validate_measurements_csv,
    return_umap_dataframe_from_quantification_dict)
import dash
from dash.exceptions import PreventUpdate
from conftest import skip_on

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
    validated_measurements, cols, err = parse_and_validate_measurements_csv(measurements_dict)
    returned_umap = return_umap_dataframe_from_quantification_dict(validated_measurements,
                cols_include=['159Tb_DCN', '160Gd_FAP', '161Dy_CLDN5', '162Dy_s6', '163Dy_mTOR', '165Ho_ER', '166Er_AR'])
    assert isinstance(returned_umap, dash_extensions.enrich.Serverside)
    assert len(pd.DataFrame(returned_umap.value)) == len(validated_measurements)
    assert return_umap_dataframe_from_quantification_dict(None) == dash.no_update
    assert return_umap_dataframe_from_quantification_dict(validated_measurements, cols_include=['bad_col']) == dash.no_update

def test_receive_alert_on_incompatible_mask():
    upload_dict = {"experiment0+++slide0+++acq0": {"channel_1": np.empty((50, 50))}}
    data_selection = "experiment0+++slide0+++acq0"
    mask_dict = {"mask": {"array": np.empty((50, 49))}}
    error, reset = send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
    assert reset is None
    assert 'error' in error.keys()
    with pytest.raises(PreventUpdate):
        # if the masks are the same, do not send error
        upload_dict = {"experiment0+++slide0+++acq0": {"channel_1": np.empty((50, 50))}}
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
    assert 'Unassigned' in dict(counts).keys()

    bounds_2 = {'xaxis.range[0]': 241, 'xaxis.range[1]': 253, 'yaxis.range[0]': -1, 'yaxis.range[1]': 4}

    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds_2,
                                                                     cell_type="new_cell_type")
    assert list(measurements["ccramic_cell_annotation"][(measurements["x_max"] == 836) &
                                                        (measurements["y_max"] == 20)]) == ['new_cell_type']
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 2
    assert len(dict(counts)) == 2
    assert 'Unassigned' in dict(counts).keys()
    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds_2,
                                                                    cell_type="new_cell_type_2")
    counts = measurements["ccramic_cell_annotation"].value_counts(normalize=True)
    assert len(dict(counts)) == 3
    assert 'new_cell_type_2' in dict(counts).keys()

def test_basic_cell_annotation_col_pop_2(get_current_dir):
    """
    Same test as above, but instead from a rectangle instead of zoom
    """
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    bounds = {'x0': 826, 'x1': 836, 'y0': 12, 'y1': 21}

    measurements = populate_cell_annotation_column_from_bounding_box(measurements, values_dict=bounds,
                                                                   cell_type="new_cell_type", box_type="rect")
    assert list(measurements["ccramic_cell_annotation"][(measurements["x_max"] == 836) &
          (measurements["y_max"] == 20)]) == ['new_cell_type']
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 1
    counts = measurements["ccramic_cell_annotation"].value_counts(normalize =True)
    assert len(dict(counts)) == 2
    assert 'Unassigned' in dict(counts).keys()

@skip_on(ValueError, "There shouldn't be a numpy truth value error on the array")
def test_convert_basic_array_to_hover_template():
    array = np.zeros((1000, 1000))
    assert len(array.shape) == 2
    template = process_mask_array_for_hovertemplate(array)
    assert template.shape[2] == 1
    assert np.unique(template) == ['None']

def test_get_cell_ids_in_svgpath(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    svgpath = 'M670.7797603577856,478.9708311618908L675.5333177884905,487.2270098573258L676.0336922548805,' \
              '492.2307545212258L671.2801348241755,500.73712044985575L669.7790114250056,' \
              '501.98805661583077L668.0277007926405,501.4876821494408L665.7760156938856,' \
              '499.2359970506858L663.5243305951306,497.9850608847108L662.2733944291556,' \
              '496.23375025234577L661.7730199627656,492.9813162208108L661.7730199627656,' \
              '491.2300055884458L662.7737688955456,490.47944388886077L665.0254539943006,' \
              '490.47944388886077L665.7760156938856,486.4764481577408L665.2756412274956,' \
              '484.72513752537577L664.7752667611055,482.7236396598158L666.0262029270806,' \
              '477.2195205295258L667.2771390930556,480.7221417942558L667.5273263262505,' \
              '481.4727034938408L668.2778880258355,479.9715800946708L668.5280752590305,479.9715800946708Z'
    cells_included_1 = get_cells_in_svg_boundary_by_mask_percentage(mask_array=mask, svgpath=svgpath)
    assert len(cells_included_1) == 2
    assert list(cells_included_1.keys()) == [403, 452]
    cells_included_2 = get_cells_in_svg_boundary_by_mask_percentage(mask_array=mask, svgpath=svgpath, threshold=0.97)
    assert len(cells_included_2) == 1
    assert list(cells_included_2.keys()) == [452]
    cells_all = get_cells_in_svg_boundary_by_mask_percentage(mask_array=mask, svgpath=svgpath, use_partial=False)
    assert len(cells_all) == 2

def test_basic_cell_annotation_col_pop_from_masking(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    svgpath = 'M670.7797603577856,478.9708311618908L675.5333177884905,487.2270098573258L676.0336922548805,' \
              '492.2307545212258L671.2801348241755,500.73712044985575L669.7790114250056,' \
              '501.98805661583077L668.0277007926405,501.4876821494408L665.7760156938856,' \
              '499.2359970506858L663.5243305951306,497.9850608847108L662.2733944291556,' \
              '496.23375025234577L661.7730199627656,492.9813162208108L661.7730199627656,' \
              '491.2300055884458L662.7737688955456,490.47944388886077L665.0254539943006,' \
              '490.47944388886077L665.7760156938856,486.4764481577408L665.2756412274956,' \
              '484.72513752537577L664.7752667611055,482.7236396598158L666.0262029270806,' \
              '477.2195205295258L667.2771390930556,480.7221417942558L667.5273263262505,' \
              '481.4727034938408L668.2778880258355,479.9715800946708L668.5280752590305,479.9715800946708Z'
    cells_included = get_cells_in_svg_boundary_by_mask_percentage(mask_array=mask, svgpath=svgpath)
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    assert "ccramic_cell_annotation" not in measurements.columns
    measurements = populate_cell_annotation_column_from_cell_id_list(measurements, cell_list=list(cells_included.keys()),
                                    cell_type="new_cell_type", sample_name="Dilution_series_1_1")
    assert "ccramic_cell_annotation" in measurements.columns
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 2
    assert list(measurements[measurements["cell_id"] == 1]["ccramic_cell_annotation"]) == ['Unassigned']
    assert list(measurements[measurements["cell_id"] == 403]["ccramic_cell_annotation"]) == ["new_cell_type"]

def test_basic_clickdata_cell_annotation(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    clickdata = {'points': [{'x': -100, 'y': -100}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new", sample="Dilution_series_1_1")
    assert 'new' not in annotations['ccramic_cell_annotation'].tolist()

    clickdata = {'points': [{'x': 53, 'y': 33}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new", sample="Dilution_series_1_1")

    assert 'new' in annotations['ccramic_cell_annotation'].tolist()
    assert dict(collections.Counter(annotations['ccramic_cell_annotation']))['new'] == 1

    clickdata = {'points': [{'x': 980, 'y': 19}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new", sample="Dilution_series_1_1")

    assert dict(collections.Counter(annotations['ccramic_cell_annotation']))['new'] == 2

    subset = subset_measurements_by_point(measurements, 53, 33)
    assert len(subset) == 1

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv")).drop(['x_min', 'y_min'],
                                                                                                 axis=1)

    clickdata = {'points': [{'x': 53, 'y': 33}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new", sample="Dilution_series_1_1")

    assert 'new' not in annotations['ccramic_cell_annotation'].tolist()

    mask = np.zeros((1000, 1000))
    mask[33, 53] = 45

    mask_dict = {"roi_1": {"raw": mask}}

    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new", sample="Dilution_series_1_1",
                                                                  mask_dict=mask_dict, mask_selection="roi_1",
                                                                  mask_toggle=True)

    assert 'new' in annotations['ccramic_cell_annotation'].tolist()





def test_generate_grid_overlay():
    """
    test that the greyscale grid overlay is generated for the correct dimensions
    """
    normal_grid = generate_greyscale_grid_array((1000, 1000))
    assert np.min(normal_grid) == 0
    assert np.max(normal_grid) == 255

    normal_grid = generate_greyscale_grid_array((75, 75))
    assert np.min(normal_grid) == 0
    assert np.max(normal_grid) == 0

def test_parse_quantification_sheet_for_roi_identifier(get_current_dir):
    """
    Test that the parser for identifying which column and value in the quantification sheet should be used
    to match the current ROI to the entries in the quantification sheet
    """
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    assert 'sample' in measurements.columns

    dataset_options = ["roi_1", "mcd1+++slide0+++Dilution_series_1_1", "roi_3"]
    data_selection = "mcd1+++slide0+++Dilution_series_1_1"
    name, column = identify_column_matching_roi_to_quantification(data_selection, measurements, dataset_options)
    assert name == "mcd1_2"
    assert column == "sample"

    measurements.rename(columns={"sample": "description"}, inplace=True)
    dataset_options = ["roi_1", "mcd1+++slide0+++Dilution_series_1_1", "roi_3"]
    data_selection = "mcd1+++slide0+++Dilution_series_1_1"
    name, column = identify_column_matching_roi_to_quantification(data_selection, measurements, dataset_options)
    assert name == "Dilution_series_1_1"
    assert column == "description"

    dataset_options = ["roi_1", "mcd1+++slide0+++roi_1", "roi_3"]
    data_selection = "mcd1+++slide0+++roi_1"
    name, column = identify_column_matching_roi_to_quantification(data_selection, measurements, dataset_options)
    assert name is None
    assert column is None
    # from steinbock
    mask_option = "test_018"
    data_selection = "test---slide0---chr10-h54h54-Gd158_2_18"
    dataset_options = ["test---slide0---chr10-h54h54-Gd158_2_18"]
    measurements = parse_quantification_sheet_from_h5ad((os.path.join(get_current_dir, "from_steinbock.h5ad")))
    name, column = identify_column_matching_roi_to_quantification(data_selection, measurements, dataset_options,
                                                                  delimiter="---", mask_name=mask_option)
    assert name == "chr10-h54h54-Gd158_2_18"
    assert column == "description"

    # from tiff, matching the experiment name
    mask_option = "MB0653_1_63_fullstack_mask"
    data_selection = "MB0653_1_63_fullstack---slide0---acq0"
    measurements = pd.DataFrame({"chan_1": [1, 2, 3, 4, 5],
                                 "description": mask_option,
                                 "cell_id": [1, 2, 3, 4, 5]})
    name, column = identify_column_matching_roi_to_quantification(data_selection, measurements, dataset_options,
                                                                  delimiter="---", mask_name=mask_option)
    assert name == mask_option

def test_annotation_column_from_umap_(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    umap_frame = pd.read_csv(os.path.join(get_current_dir, "umap_coordinates_for_measurements.csv"))

    layout = {'xaxis.range[0]': 7.73432730097818, 'xaxis.range[1]': 9.547373230248308,
                        'yaxis.range[0]': 7.02148605737705, 'yaxis.range[1]': 9.10655368032787}

    indices_in = [16, 23, 26, 29, 30, 57, 63, 64, 67, 82, 87, 89, 138, 153, 157, 160,
                  172, 178, 197, 201, 205, 206, 215, 219, 225, 229, 234, 235, 237, 239]

    measurements = populate_quantification_frame_column_from_umap_subsetting(measurements, umap_frame, layout,
                                                                             annotation_column="broad_class",
                                                                             annotation_value="test_cell_type")
    for index in range(len(measurements)):
        if index in indices_in:
            assert measurements["broad_class"].tolist()[index] == "test_cell_type"
        else:
            assert measurements["broad_class"].tolist()[index] == 'Unassigned'


def test_apply_cluster_annotations_to_mask(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    cluster_dict = {'Type_1': '#932652', 'Type_2': '#FAE4B0', 'Type_3': '#DCCAFC'}
    cluster_assignments = pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))
    with_annotations = generate_mask_with_cluster_annotations(mask, cluster_assignments, cluster_dict)
    assert np.array_equal(with_annotations[532, 457], np.array([250, 228, 176]))
    assert with_annotations.shape == (mask.shape[0], mask.shape[1], 3)
    # assert where type 1 is
    assert list(with_annotations[449, 414]) == list(with_annotations[484, 852]) == [147, 38, 82]
    # assert where no cells are
    assert list(with_annotations[623, 420]) == list(with_annotations[787, 709]) == [0, 0, 0]
    # assert where there are cells that are not annotated (remain as white)
    assert list(with_annotations[864, 429]) == list(with_annotations[784, 799]) == [255, 255, 255]

    # run without keeping the cells that are not annotated
    with_annotations = generate_mask_with_cluster_annotations(mask, cluster_assignments, cluster_dict,
                                                              retain_cells=False)
    # assert that where cells were before, there is nothing
    assert list(with_annotations[864, 429]) == list(with_annotations[784, 799]) == [0, 0, 0]

def test_apply_cluster_annotations_with_gating(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    cluster_dict = {'Type_1': '#932652', 'Type_2': '#FAE4B0', 'Type_3': '#DCCAFC'}
    cluster_assignments = pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))
    gating_list = list(range(125, 175))
    with_annotations = generate_mask_with_cluster_annotations(mask, cluster_assignments, cluster_dict,
                                                              use_gating_subset=True, gating_subset_list=gating_list)
    # assert that a position where the cell is not gated, is blank
    assert np.array_equal(with_annotations[532, 457], np.array([0, 0, 0]))
    fake_frame = pd.DataFrame({"missing_key": [1, 2, 3, 4, 5],
                               "cluster": ["immune"] * 5})
    assert generate_mask_with_cluster_annotations(mask, cluster_assignments, fake_frame,
        use_gating_subset=True, gating_subset_list=gating_list) is None


def test_remove_latest_annotation():
    annotations_dict_original = {"roi_1": {"annot_1": "This is an annotation", "annot_2": "This is also an annotation"}}
    annotations_dict = remove_annotation_entry_by_indices(annotations_dict_original, "roi_1")
    assert len(annotations_dict['roi_1']) == 1
    annotations_dict = remove_annotation_entry_by_indices(annotations_dict, "roi_1")
    assert not len(annotations_dict['roi_1'])
    annotations_dict = remove_annotation_entry_by_indices(annotations_dict, "roi_1")
    assert not len(annotations_dict['roi_1'])
    assert remove_annotation_entry_by_indices(annotations_dict, None) == annotations_dict_original
    assert not remove_annotation_entry_by_indices(None, "roi_1")

def test_quantification_distribution_table(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    dist = pd.DataFrame(quantification_distribution_table(measurements, umap_variable="sample"))
    assert 'Proportion' in dist.columns
    assert dist['Proportion'].to_list() == [0.828, 0.172]
    subset = {'test_1': 97, 'test_2': 8}
    dist = pd.DataFrame(quantification_distribution_table(measurements, umap_variable="sample",
                                                          subset_cur_cat=subset))
    assert dist['Proportion'].to_list() == [0.924, 0.076]


def test_custom_gating_id_from_string():
    assert custom_gating_id_list() == []
    gating_list = custom_gating_id_list("130,145,156, 170  ,   not_a_number")
    assert len(gating_list) == 4
    assert all([isinstance(elem, int) for elem in gating_list])
