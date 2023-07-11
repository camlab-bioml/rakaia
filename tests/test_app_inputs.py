from dash_extensions.enrich import dcc

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
