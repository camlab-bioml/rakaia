from ccramic.app.inputs.pixel_level_inputs import *
import numpy as np

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
    with_scaleval = add_scale_value_to_figure(image, recoloured.shape, 0.05)
    assert len(with_scaleval['layout']['annotations']) != 0
    assert with_scaleval['layout']['annotations'][0]['text'] == '<span style="color: white">45μm</span><br>'

    custom_scaleval = add_scale_value_to_figure(image, recoloured.shape, 0.05, scale_value=51)
    assert custom_scaleval['layout']['annotations'][0]['text'] == '<span style="color: white">51μm</span><br>'
