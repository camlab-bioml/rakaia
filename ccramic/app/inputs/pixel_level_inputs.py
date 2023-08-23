import dash
from dash_extensions.enrich import dcc
from ..parsers.pixel_level_parsers import *
from ..utils.cell_level_utils import *
import plotly.graph_objs as go
import dash_draggable
import math
import cv2
from plotly.graph_objs.layout import XAxis, YAxis

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

def wrap_canvas_in_loading_screen_for_large_images(image=None, size_threshold=3000, hovertext=False):
    """
    Wrap the annotation canvas in a dcc.Loading screen if the dimensions of the image are larger than the threshold
    or
    if hovertext is used (slows down the canvas considerably)
    """
    # conditions for wrapping the canvas
    large_image = image is not None and (image.shape[0] > size_threshold or image.shape[1] > size_threshold)
    if large_image or hovertext:
        return dcc.Loading(render_default_annotation_canvas(),
                                     type="default", fullscreen=False)
    else:
        return render_default_annotation_canvas()

def add_scale_value_to_figure(figure, image_shape, scale_value=None, font_size=12, x_axis_left=0.05):
    """
    add a scalebar value to a canvas figure based on the dimensions of the current image
    """
    if scale_value is None:
        scale_val = int(0.075 * image_shape[1])
    else:
        scale_val = scale_value
    scale_annot = str(scale_val) + "μm"
    scale_text = f'<span style="color: white">{scale_annot}</span><br>'
    figure = go.Figure(figure)
    # the midpoint of the annotation is set by the middle of 0.05 and 0.125 and an xanchor of center`
    figure.add_annotation(text=scale_text, font={"size": font_size}, xref='paper',
                       yref='paper',
                       # set the placement of where the text goes relative to the scale bar
                       x=float(x_axis_left + 0.0375),
                       xanchor='center',
                       y=0.06,
                       # yanchor='bottom',
                       showarrow=False)
    return figure


def get_additive_image_with_masking(currently_selected, data_selection, canvas_layers, mask_config,
                                    mask_toggle, mask_selection, show_canvas_legend,
                                    mask_blending_level, add_mask_boundary, legend_text, annotation_size=12):
    """
    Generate an additiive image from one or more channel arrays. Optionally, project a mask on top of the additive image
    using a specified blend ratio with cv2
    """
    split = split_string_at_pattern(data_selection)
    exp, slide, acq = split[0], split[1], split[2]
    try:
        image = sum([np.asarray(canvas_layers[exp][slide][acq][elem]).astype(np.float32) for \
                 elem in currently_selected if \
                 elem in canvas_layers[exp][slide][acq].keys()]).astype(np.float32)
        image = np.clip(image, 0, 255)
        if mask_toggle and None not in (mask_config, mask_selection) and len(mask_config) > 0:
            if image.shape[0] == mask_config[mask_selection].shape[0] and \
                image.shape[1] == mask_config[mask_selection].shape[1]:
                # set the mask blending level based on the slider, by default use an equal blend
                mask_level = float(mask_blending_level / 100) if mask_blending_level is not None else 1
                image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    mask_config[mask_selection].astype(np.uint8), mask_level, 0)
                if add_mask_boundary:
                    # add the border of the mask after converting back to greyscale to derive the conversion
                    greyscale_mask = np.array(Image.fromarray(mask_config[mask_selection]).convert('L'))
                    reconverted = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(greyscale_mask)).convert('RGB'))
                    image = cv2.addWeighted(image.astype(np.uint8), 1, reconverted.astype(np.uint8), 1, 0)
        default_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"
        fig = px.imshow(Image.fromarray(image.astype(np.uint8)))
        image_shape = image.shape
        if show_canvas_legend:
            x_axis_placement = 0.00001 * image_shape[1]
            # make sure the placement is min 0.05 and max 0.1
            x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.1 else 0.05
            # if the current graph already has an image, take the existing layout and apply it to the new figure
            # otherwise, set the uirevision for the first time
            fig = add_scale_value_to_figure(fig, image_shape, font_size=annotation_size)

            if legend_text != '' and show_canvas_legend:
                fig.add_annotation(text=legend_text, font={"size": (annotation_size + 3)}, xref='paper',
                               yref='paper',
                               x=(1 - x_axis_placement),
                               # xanchor='right',
                               y=0.05,
                               # yanchor='bottom',
                               bgcolor="black",
                               showarrow=False)

            # set the x-axis scale placement based on the size of the image
            # for adding a scale bar
            if show_canvas_legend:
                fig.add_shape(type="line",
                          xref="paper", yref="paper",
                          x0=x_axis_placement, y0=0.05, x1=(x_axis_placement + 0.075),
                          y1=0.05,
                          line=dict(
                              color="white",
                              width=2,
                          ))

        fig['layout']['uirevision'] = True
        fig.update_traces(hovertemplate=default_hover)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis=XAxis(showticklabels=False, domain=[0, 1]),
                      yaxis=YAxis(showticklabels=False),
                      margin=dict(
                          l=10,
                          r=0,
                          b=25,
                          t=35,
                          pad=0
                      ))
        fig.update_layout(hovermode="x")
        return fig
    except KeyError:
        return dash.no_update
