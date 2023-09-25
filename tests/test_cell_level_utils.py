import collections

import numpy as np
import pytest
from ccramic.app.utils.cell_level_utils import *
from ccramic.app.parsers.cell_level_parsers import *
from ccramic.app.inputs.cell_level_inputs import *
from ccramic.app.io.annotation_outputs import *
import os
from PIL import Image
import dash_extensions
import tempfile

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
    returned_umap = return_umap_dataframe_from_quantification_dict(validated_measurements)
    assert isinstance(returned_umap, tuple)
    assert isinstance(returned_umap[0], dash_extensions.enrich.Serverside)
    assert isinstance(returned_umap[1], list)
    with pytest.raises(PreventUpdate):
        return_umap_dataframe_from_quantification_dict(None)

def test_receive_alert_on_incompatible_mask():
    upload_dict = {"experiment0+++slide0+++acq0": {"channel_1": np.empty((50, 50))}}
    data_selection = "experiment0+++slide0+++acq0"
    mask_dict = {"mask": {"array": np.empty((50, 49))}}
    error = send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, None, "mask", True)
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
    assert 'None' in dict(counts).keys()


def test_convert_basic_array_to_hovertemplate():
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
                                                                     cell_type="new_cell_type")
    assert "ccramic_cell_annotation" in measurements.columns
    assert len(measurements[measurements["ccramic_cell_annotation"] == "new_cell_type"]) == 2
    assert list(measurements[measurements["cell_id"] == 1]["ccramic_cell_annotation"]) == ["None"]
    assert list(measurements[measurements["cell_id"] == 403]["ccramic_cell_annotation"]) == ["new_cell_type"]


def test_output_annotations_pdf():
    """
    test that the output of the annotations pdf produces a valid file
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "ccramic_test_annotations.pdf")
        assert not os.path.exists(file_path)
        data_selection = "exp1+++slide0+++roi_1"
        range_tuple = tuple(sorted({'xaxis.range[0]': 50, 'xaxis.range[1]': 100,
                    'yaxis.range[0]': 50, 'yaxis.range[1]': 100}.items()))
        annotations_dict = {data_selection: {range_tuple: {'title': 'Title', 'body': 'body',
                                                               'cell_type': 'cell_type', 'imported': False,
                                                            'type': 'zoom', 'channels': ['channel_1'],
                                                             'use_mask': False,
                                                             'mask_selection': None,
                                                             'mask_blending_level': 35,
                                                             'add_mask_boundary': False}}}
        layers_dict = {"exp1+++slide0+++roi_1":
                           {"channel_1": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB'))}}
        aliases = {"channel_1": "channel_1"}
        output_pdf = generate_annotations_output_pdf(annotations_dict, layers_dict, data_selection,
                                                     mask_config=None, aliases=aliases, dest_dir=tmpdirname)
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)

        mask_config = {"mask": {"array": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB')),
                                "raw": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB')),
                                "boundary": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB'))}}
        annotations_dict = {data_selection: {range_tuple: {'title': 'Title', 'body': 'body',
                                                           'cell_type': 'cell_type', 'imported': False,
                                                           'type': 'zoom', 'channels': ['channel_1'],
                                                           'use_mask': True,
                                                           'mask_selection': "mask",
                                                           'mask_blending_level': 35,
                                                           'add_mask_boundary': True}}}

        output_pdf = generate_annotations_output_pdf(annotations_dict, layers_dict, data_selection,
                                                     mask_config=mask_config, aliases=aliases, dest_dir=tmpdirname)
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(file_path)

        blend_dict = {"channel_1": {'color': '#FFFFFF'}}

        output_pdf = generate_annotations_output_pdf(annotations_dict, layers_dict, data_selection,
                                                     mask_config=mask_config, aliases=aliases, dest_dir=tmpdirname,
                                                     blend_dict=blend_dict)
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)


def test_basic_clickdata_cell_annotation(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    clickdata = {'points': [{'x': -100, 'y': -100}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new")
    assert 'new' not in annotations['ccramic_cell_annotation'].tolist()

    clickdata = {'points': [{'x': 53, 'y': 33}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new")

    assert 'new' in annotations['ccramic_cell_annotation'].tolist()
    assert dict(collections.Counter(annotations['ccramic_cell_annotation']))['new'] == 1

    clickdata = {'points': [{'x': 980, 'y': 19}]}
    annotations = populate_cell_annotation_column_from_clickpoint(measurements, None, values_dict=clickdata,
                                                                  cell_type="new")

    assert dict(collections.Counter(annotations['ccramic_cell_annotation']))['new'] == 2

    subset = subset_measurements_by_point(measurements, 53, 33)
    assert len(subset) == 1


def test_output_annotations_masks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        annotations_dict = {'Patient1+++slide0+++pos1_1': {(('xaxis.range[0]', 384.3802395209581),
                                                        ('xaxis.range[1]', 487.6736526946108),
                                                        ('yaxis.range[0]', 426.1467065868263),
                                                        ('yaxis.range[1]', 322.8532934131736)): {'title': 'test',
                                                                                                 'body': 'test',
                                                                                                 'cell_type': 'cell type 1',
                                                                                                 'imported': False,
                                                                                                 'annotation_column': 'ccramic_cell_annotation',
                                                                                                 'type': 'zoom',
                                                                                                 'channels': ['Ho165'],
                                                                                                 'use_mask': None,
                                                                                                 'mask_selection': None,
                                                                                                 'mask_blending_level': 35,
                                                                                                 'add_mask_boundary': [
                                                                                                     ' add boundary']},
                                                       'M216.41616766467067,157.58383233532933L235.27844311377245,185.42814371257487L240.6676646706587,210.57784431137725L241.56586826347305,239.32035928143713L241.56586826347305,254.58982035928145L233.48203592814372,270.75748502994014L207.43413173652695,293.2125748502994L189.47005988023952,299.5L161.625748502994,297.7035928143713L143.66167664670658,290.5179640718563L129.29041916167665,275.248502994012L119.41017964071857,256.3862275449102L117.61377245508982,224.94910179640718L132.88323353293413,188.12275449101796L143.66167664670658,186.32634730538922L174.2005988023952,185.42814371257487L179.58982035928145,166.56586826347305L184.0808383233533,154.88922155688624L185.87724550898204,153.99101796407186Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 2',
                                                           'imported': False,
                                                           'annotation_column': 'ccramic_cell_annotation',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']}, (
                                                       ('x0', 198.45209580838323), ('x1', 440.9670658682635),
                                                       ('y0', 40.81736526946108), ('y1', 155.7874251497006)): {
            'title': 'test', 'body': 'test', 'cell_type': 'cell type 3', 'imported': False,
            'annotation_column': 'broad', 'type': 'rect', 'channels': ['Ho165'], 'use_mask': None,
            'mask_selection': None, 'mask_blending_level': 35, 'add_mask_boundary': [' add boundary']},
                                                       'M97.85329341317365,422.55389221556885L114.02095808383234,431.53592814371257L136.47604790419163,456.685628742515L164.32035928143713,500.69760479041923L168.811377245509,514.1706586826348L167.9131736526946,533.9311377245509L159.82934131736528,541.116766467066L127.4940119760479,542.9131736526947L113.12275449101796,538.4221556886229L90.66766467065868,524.0508982035929L61.026946107784426,500.69760479041923L40.368263473053894,470.1586826347306L34.97904191616767,453.0928143712575L34.97904191616767,434.2305389221557L53.84131736526947,423.45209580838326L54.73952095808384,423.45209580838326Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 4',
                                                           'imported': False, 'annotation_column': 'broad',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']}}}

        assert not os.path.exists(os.path.join(tmpdirname, "annotation_masks.zip"))
        output_dir = export_annotations_as_masks(annotations_dict, tmpdirname,
                                                 'Patient1+++slide0+++pos1_1', (600, 600, 1))
        assert os.path.exists(os.path.join(output_dir))
