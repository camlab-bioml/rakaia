from ccramic.app.inputs.pixel_level_inputs import *
import numpy as np
import plotly.graph_objs as go
from dash_extensions.enrich import html

def test_return_canvas_input():
    default_graph = render_default_annotation_canvas()
    assert isinstance(default_graph, dcc.Graph)
    assert not default_graph.config['scrollZoom']
    fullscreen_graph = render_default_annotation_canvas(fullscreen_mode=True)
    assert fullscreen_graph.config['scrollZoom']

def test_wrapping_canvas_based_on_image_dimensions():
    small_image = np.zeros((512,512,3), 'uint8')
    small_canvas = wrap_canvas_in_loading_screen_for_large_images(small_image)
    assert isinstance(small_canvas, dcc.Graph)
    large_image = np.zeros((3001,3001,3), 'uint8')
    large_canvas = wrap_canvas_in_loading_screen_for_large_images(large_image)
    assert not isinstance(large_canvas, dcc.Graph)
    assert isinstance(large_canvas, dcc.Loading)

def test_add_scalebar_to_canvas(get_current_dir):
    greyscale = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    recoloured = recolour_greyscale(np.array(greyscale), colour='#D14A1A')
    image = go.Figure(px.imshow(recoloured))
    assert len(image['layout']['annotations']) == 0
    with_scaleval = add_scale_value_to_figure(image, recoloured.shape)
    assert len(with_scaleval['layout']['annotations']) != 0
    assert with_scaleval['layout']['annotations'][0]['text'] == '<span style="color: white">45μm</span><br>'

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
                                            add_mask_boundary=False, legend_text='')
    assert isinstance(image, go.Figure)
    assert image['data'] is not None
    assert image['data'][0]['hovertemplate'] == 'x: %{x}<br>y: %{y}<br><extra></extra>'
    assert image['layout']['annotations'][0]['text'] == '<span style="color: white">45μm</span><br>'
    assert image['layout']['uirevision']


    mask_config = {"mask": np.zeros((600, 600, 3)).astype(np.uint8)}

    image_mask = get_additive_image_with_masking(["DNA", "Nuclear"], data_selection="experiment0+++slide0+++acq0",
                                            canvas_layers=upload_dict, mask_config=mask_config, mask_toggle=True,
                                            mask_selection="mask", show_canvas_legend=True, mask_blending_level=1,
                                            add_mask_boundary=True, legend_text='')
    assert isinstance(image_mask, go.Figure)
    assert image_mask['data'] is not None
    assert image_mask['data'][0]['hovertemplate'] == 'x: %{x}<br>y: %{y}<br><extra></extra>'
    assert image_mask['layout']['annotations'][0]['text'] == '<span style="color: white">45μm</span><br>'
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

    assert image['layout']['annotations'][0]['x'] == 0.0875
    assert image['layout']['shapes'][0]['x0'] == 0.05

    image = invert_annotations_figure(image)
    assert image['layout']['annotations'][0]['x'] == (1 - 0.0875)
    assert image['layout']['shapes'][0]['x0'] == (1 - 0.05)
