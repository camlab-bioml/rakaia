
from dash_extensions.enrich import dcc
from ..parsers.pixel_level_parsers import *
import plotly.graph_objs as go
import dash_draggable
import math

def render_default_annotation_canvas(input_id: str="annotation_canvas", fullscreen_mode=False,
                                     draggable=False):
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

    canvas = dcc.Graph(config={"modeBarButtonsToAdd": [
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

    return dash_draggable.GridLayout(id='draggable', children=[canvas]) if draggable else canvas

def wrap_canvas_in_loading_screen_for_large_images(image, size_threshold=3000):
    """
    Wrap the annotation canvas in a dcc.Loading screen if the dimensions of the image are larger than the threshold
    """
    if image.shape[0] > size_threshold or image.shape[1] > size_threshold:
        return dcc.Loading(render_default_annotation_canvas(),
                                     type="default", fullscreen=False)
    else:
        return render_default_annotation_canvas()


def add_scale_value_to_figure(figure, image_shape, x_axis_placement, scale_value=None):
    """
    add a scalebar value to a canvas figure based on the dimensions of the current image
    """
    if scale_value is None:
        scale_val = int(0.075 * image_shape[1])
    else:
        scale_val = scale_value
    scale_annot = str(scale_val) + "μm"
    scale_text = f'<span style="color: white">{scale_annot}</span><br>'
    # this is the middle point of the scale bar
    # add shift based on the image shape
    shift = math.log10(image_shape[1]) - 3
    midpoint = (x_axis_placement + (0.075 / (2.5 * len(str(scale_val)) + shift)))
    # ensure that the text label does not go beyond the scale bar or over the midpoint of the scale bar
    midpoint = midpoint if (0.05 < midpoint < 0.0875) else x_axis_placement
    font_size = 10 if image_shape[1] < 1000 else 12
    midpoint = midpoint if font_size == 12 else 0.05
    figure = go.Figure(figure)
    figure.add_annotation(text=scale_text, font={"size": font_size}, xref='paper',
                       yref='paper',
                       # set the placement of where the text goes relative to the scale bar
                       x=midpoint,
                       # xanchor='right',
                       y=0.06,
                       # yanchor='bottom',
                       showarrow=False)
    return figure
