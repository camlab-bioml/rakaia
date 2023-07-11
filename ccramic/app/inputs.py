
from dash_extensions.enrich import dcc
from .parsers import *

def render_default_annotation_canvas(input_id: str="annotation_canvas", fullscreen_mode=False):
    """
    Return the default dcc.Graph annotation figure input. For multiple annotation graphs, a unique input ID
    must be used
    """

    # select the style based on if the canvas is fullscreen or partial

    if fullscreen_mode:
        style_canvas = {"margin": "auto", "width": "100vw", "height": "100vh",
                   "max-width": "none", "max-height": "none"}
    else:
        style_canvas = {"width": "65vw", "height": "65vh"}

    return dcc.Graph(config={"modeBarButtonsToAdd": [
                        # "drawline",
                        # "drawopenpath",
                        "drawclosedpath",
                        # "drawcircle",
                        "drawrect",
                        "eraseshape"],
                        'toImageButtonOptions': {'format': 'png', 'filename': 'canvas', 'scale': 1},
                            # disable scrollable zoom for now to control the scale bar
                        'edits': {'shapePosition': False}, 'scrollZoom': fullscreen_mode},
                        relayoutData={'autosize': True},
                        id=input_id,
                            style=style_canvas,
                        figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                              xaxis=go.XAxis(showticklabels=False),
                                              yaxis=go.YAxis(showticklabels=False))})


def wrap_canvas_in_loading_screen_for_large_images(image, size_threshold=3000):
    """
    Wrap the annotation canvas in a dcc.Loading screen if the dimensions of the image are larger than the threshold
    """
    if image.shape[0] > size_threshold or image.shape[1] > size_threshold:
        return dcc.Loading(render_default_annotation_canvas(),
                                     type="default", fullscreen=False)
    else:
        return render_default_annotation_canvas()
