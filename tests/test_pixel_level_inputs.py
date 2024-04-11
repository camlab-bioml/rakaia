import dash
import numpy as np
import plotly.graph_objs as go
import pytest
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import html
from ccramic.inputs.pixel_level_inputs import (
    render_default_annotation_canvas,
    wrap_canvas_in_loading_screen_for_large_images,
    add_scale_value_to_figure,
    get_additive_image_with_masking,
    add_local_file_dialog,
    invert_annotations_figure,
    set_range_slider_tick_markers,
    generate_canvas_legend_text,
    set_x_axis_placement_of_scalebar, update_canvas_filename,
    set_canvas_viewport,
    marker_correlation_children,
    reset_pixel_histogram)
from ccramic.parsers.pixel_level_parsers import create_new_blending_dict
import dash_core_components as dcc
from PIL import Image
import os
from ccramic.utils.pixel_level_utils import recolour_greyscale
import plotly.express as px
import dash_bootstrap_components as dbc

def test_return_canvas_input():
    default_graph = render_default_annotation_canvas()
    assert isinstance(default_graph, dcc.Graph)
    assert not default_graph.config['scrollZoom']
    fullscreen_graph = render_default_annotation_canvas(fullscreen_mode=True)
    assert fullscreen_graph.config['scrollZoom']

def test_wrapping_canvas_based_on_image_dimensions():
    small_image = np.zeros((512,512,3), 'uint8')
    small_canvas = wrap_canvas_in_loading_screen_for_large_images(small_image, filename="exp0+++slide0+++roi_1")
    assert isinstance(small_canvas, dcc.Graph)
    large_image = np.zeros((3001,3001,3), 'uint8')
    large_canvas = wrap_canvas_in_loading_screen_for_large_images(large_image, filename="canvas_split")
    assert not isinstance(large_canvas, dcc.Graph)
    assert isinstance(large_canvas, dcc.Loading)

def test_add_scalebar_to_canvas(get_current_dir):
    greyscale = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    recoloured = recolour_greyscale(np.array(greyscale), colour='#D14A1A')
    image = go.Figure(px.imshow(recoloured))
    assert len(image['layout']['annotations']) == 0
    with_scaleval = add_scale_value_to_figure(image, recoloured.shape)
    assert len(with_scaleval['layout']['annotations']) != 0
    assert with_scaleval['layout']['annotations'][0]['text'] == '<span style="color: white">60μm</span><br>'

    custom_scaleval = add_scale_value_to_figure(image, recoloured.shape, scale_value=51)
    assert custom_scaleval['layout']['annotations'][0]['text'] == '<span style="color: white">51μm</span><br>'


def test_basic_additive_image():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.zeros((600, 600, 3)),
                                                       "Nuclear": np.zeros((600, 600, 3)),
                                                       "Cytoplasm": np.zeros((600, 600, 3))},
                                              "experiment0+++slide0+++acq1": {"DNA": np.zeros((600, 600, 3)),
                                                       "Nuclear": np.zeros((600, 600, 3)),
                                                       "Cytoplasm": np.zeros((600, 600, 3))}
                                              }

    # blend_dict = create_new_blending_dict(upload_dict)

    image = get_additive_image_with_masking(["DNA", "Nuclear"], data_selection="experiment0+++slide0+++acq0",
                                            canvas_layers=upload_dict, mask_config=None, mask_toggle=False,
                                            mask_selection=None, show_canvas_legend=True, mask_blending_level=1,
                                            add_mask_boundary=False, legend_text='this is a legend')
    assert isinstance(image, go.Figure)
    assert image['data'] is not None
    assert image['data'][0]['hovertemplate'] == 'x: %{x}<br>y: %{y}<br><extra></extra>'
    assert image['layout']['annotations'][0]['text'] == '<span style="color: white">60μm</span><br>'
    assert image['layout']['uirevision']

    bad_col = get_additive_image_with_masking(["fake_channel"], data_selection="experiment0+++slide0+++acq0",
                                        canvas_layers=upload_dict, mask_config=None, mask_toggle=False,
                                        mask_selection=None, show_canvas_legend=True, mask_blending_level=1,
                                        add_mask_boundary=False, legend_text='this is a legend')

    assert isinstance(bad_col, dash._callback.NoUpdate)


    mask_config = {"mask": np.zeros((600, 600, 3)).astype(np.uint8)}

    image_mask = get_additive_image_with_masking(["DNA", "Nuclear"], data_selection="experiment0+++slide0+++acq0",
                                            canvas_layers=upload_dict, mask_config=mask_config, mask_toggle=True,
                                            mask_selection="mask", show_canvas_legend=True, mask_blending_level=1,
                                            add_mask_boundary=True, legend_text='')
    assert isinstance(image_mask, go.Figure)
    assert image_mask['data'] is not None
    assert image_mask['data'][0]['hovertemplate'] == 'x: %{x}<br>y: %{y}<br><extra></extra>'
    assert image_mask['layout']['annotations'][0]['text'] == '<span style="color: white">60μm</span><br>'
    assert image_mask['layout']['uirevision']


def test_basic_return_local_file_dialog():
    assert isinstance(add_local_file_dialog(use_local_dialog=True), dbc.Button)
    assert isinstance(add_local_file_dialog(use_local_dialog=False), html.Div)

def test_invert_annotations_figure():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.zeros((600, 600, 3)),
                                                   "Nuclear": np.zeros((600, 600, 3)),
                                                   "Cytoplasm": np.zeros((600, 600, 3))},
                   "experiment0+++slide0+++acq1": {"DNA": np.zeros((600, 600, 3)),
                                                   "Nuclear": np.zeros((600, 600, 3)),
                                                   "Cytoplasm": np.zeros((600, 600, 3))}
                   }

    # blend_dict = create_new_blending_dict(upload_dict)

    image = get_additive_image_with_masking(["DNA", "Nuclear"], data_selection="experiment0+++slide0+++acq0",
                                            canvas_layers=upload_dict, mask_config=None, mask_toggle=False,
                                            mask_selection=None, show_canvas_legend=True, mask_blending_level=1,
                                            add_mask_boundary=False, legend_text='')

    assert image['layout']['annotations'][0]['x'] == 0.1
    assert image['layout']['shapes'][0]['x0'] == 0.05

    image = invert_annotations_figure(image)
    assert image['layout']['annotations'][0]['x'] == (1 - 0.1)
    assert image['layout']['shapes'][0]['x0'] == (1 - 0.05)

    image_2 = go.Figure()
    image_2 = invert_annotations_figure(image_2)
    assert len(image_2['layout']['annotations']) == 0


def test_tick_marker_spacing_range_slider():

    assert len(set_range_slider_tick_markers(3)[0]) == 4
    assert len(set_range_slider_tick_markers(2)[0]) == 3
    assert set_range_slider_tick_markers(100)[0] == {0: '0', 33: '33', 66: '66', 100: '100'}
    low_range, small_step = set_range_slider_tick_markers(0.3)
    assert len(low_range) == 2
    assert small_step == 0.03


def test_generate_legend_text_channels():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                   "Nuclear": np.array([1, 1, 1, 1]),
                                                   "Cytoplasm": np.array([2, 2, 2, 2]),
                                                   "Other_Nuclear": np.array([3, 3, 3, 3])},
                   "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                   "Nuclear": np.array([4, 4, 4, 4]),
                                                   "Cytoplasm": np.array([5, 5, 5, 5]),
                                                   "Other_Nuclear": np.array([6, 6, 6, 6])}}
    blend_dict = create_new_blending_dict(upload_dict)
    channel_order = list(blend_dict.keys())
    aliases = {"DNA": "dna", "Nuclear": "nuclear", "Cytoplasm": "cyto", "Other_Nuclear": "nuclear"}
    orientation = "horizontal"
    legend_text = generate_canvas_legend_text(blend_dict, channel_order, aliases, orientation)
    assert "<br>" not in legend_text
    assert "dna" in legend_text
    assert not "DNA" in legend_text
    legend_text = generate_canvas_legend_text(blend_dict, channel_order, aliases, "vertical")
    assert "<br>" in legend_text
    assert "dna" in legend_text
    assert not "DNA" in legend_text

    # assert each alias shows up only once in the legend
    assert legend_text == '<span style="color:#FFFFFF">dna</span><br><span style="color:#FFFFFF">' \
                          'nuclear</span><br><span style="color:#FFFFFF">cyto</span><br>'


def test_generate_legend_text_clustering():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                   "Nuclear": np.array([1, 1, 1, 1]),
                                                   "Cytoplasm": np.array([2, 2, 2, 2])},
                   "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                   "Nuclear": np.array([4, 4, 4, 4]),
                                                   "Cytoplasm": np.array([5, 5, 5, 5])}
                   }
    blend_dict = create_new_blending_dict(upload_dict)
    channel_order = list(blend_dict.keys())
    aliases = {"DNA": "dna", "Nuclear": "nuclear", "Cytoplasm": "cyto"}
    annot_dict = {"experiment0+++slide0+++acq0": {"cell_type_1": '#00FF66', "cell_type_2": "5500FF",
                                                  "cell_type_3": "FF009A"}}
    legend_text = generate_canvas_legend_text(blend_dict, channel_order, aliases, "vertical",
                                              True, annot_dict, "experiment0+++slide0+++acq0")
    assert legend_text == '<span style="color:#00FF66">cell_type_1</span><br><span style="color:5500FF">' \
                          'cell_type_2</span><br><span style="color:FF009A">cell_type_3</span><br>'
    assert not generate_canvas_legend_text(blend_dict, channel_order, aliases, "vertical",
                                              True, annot_dict, "experiment0+++slide0+++acq1")


def test_register_x_axis_placement_scalebar():
    image = np.zeros((1500, 1500))
    placement = set_x_axis_placement_of_scalebar(image.shape[1], False)
    assert placement == 0.05
    invert = set_x_axis_placement_of_scalebar(image.shape[1], True)
    assert invert == 0.95

    large = np.zeros((5260, 5260))
    placement = set_x_axis_placement_of_scalebar(large.shape[1], False)
    assert placement == 0.000025 * large.shape[1]

def test_set_canvas_filename():
    canvas_config = {"modeBarButtonsToAdd": ["drawclosedpath", "drawrect", "eraseshape"],
                        'toImageButtonOptions': {'format': 'png', 'filename': "canvas", 'scale': 1},
                            # disable scrollable zoom for now to control the scale bar
                        'edits': {'shapePosition': False}, 'scrollZoom': True}
    canvas_config = update_canvas_filename(canvas_config, "exp0+++slide0+++long_roi")
    assert canvas_config['toImageButtonOptions']['filename'] == "long_roi"
    canvas_config = update_canvas_filename(canvas_config, "exp0---slide0---roi_1")
    assert canvas_config['toImageButtonOptions']['filename'] == "exp0---slide0---roi_1"
    # use the filename (experiment) by default if the roi name is not long enough
    canvas_config = update_canvas_filename(canvas_config, "exp0---slide0---roi_1", delimiter='---')
    assert canvas_config['toImageButtonOptions']['filename'] == "exp0"
    assert update_canvas_filename({"fake_dict": None}, "exp0+++slide0+++long_roi") == {"fake_dict": None}


def test_window_viewport_settings():
    image_dict = {"roi_1": {"channel_1": np.zeros((2500, 1000))}}
    cur_layout = {'height': None, 'width': None}
    viewport = set_canvas_viewport(30, image_dict, "roi_1", None, cur_layout)
    assert viewport == {'width': '12.0vh', 'height': '30.0vh'}
    with pytest.raises(PreventUpdate):
        set_canvas_viewport(30, image_dict, "roi_1", None, viewport)
    assert set_canvas_viewport(30, image_dict, "roi_1", None, {}) == {'width': '12.0vh', 'height': '30.0vh'}

    blank_image_dict = {"roi_1": {}}
    canvas_layout = {'layout': {'xaxis': {"range": [0, 1000]}, 'yaxis': {"range": [2500, 0]}}}
    assert set_canvas_viewport(30, blank_image_dict, "roi_1", canvas_layout, cur_layout) == \
           {'width': '12.0vh', 'height': '30.0vh'}

    assert set_canvas_viewport(30, blank_image_dict, "roi_1", {}, {}) == \
           {'width': '30.0vh', 'height': '30.0vh'}

def test_generate_marker_correlation_information():
    children = marker_correlation_children(None, None)
    assert not children
    children = marker_correlation_children(0.50, 1.00, 1.00)
    span_counts = 0
    for child in children:
        if isinstance(child, html.Span):
            span_counts += 1
    # assert one span for each of the values
    assert span_counts == 3
    assert len(children) > 4
    assert children

def test_blank_reset_histogram():
    blank_hist = reset_pixel_histogram(True)
    assert blank_hist['layout']['margin'] == {'b': 15, 'l': 5, 'pad': 0, 'r': 5, 't': 20}
    assert not blank_hist['layout']['xaxis']['showticklabels']
