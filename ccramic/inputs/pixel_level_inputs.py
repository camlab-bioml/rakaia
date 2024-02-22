import dash
from dash_extensions.enrich import dcc, html
from ccramic.utils.cell_level_utils import convert_mask_to_cell_boundary
import plotly.graph_objs as go
import dash_draggable
import cv2
from plotly.graph_objs.layout import XAxis, YAxis
import dash_bootstrap_components as dbc
from copy import deepcopy
import numpy as np
from PIL import Image
import plotly.express as px
from ccramic.io.session import SessionTheme
from ccramic.utils.pixel_level_utils import split_string_at_pattern


def render_default_annotation_canvas(input_id: str="annotation_canvas", fullscreen_mode=False,
                                     draggable=False, filename: str="canvas", delimiter: str="+++"):
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

    # set a custom output filename based on the current ROI
    if filename != "canvas":
        filename = set_roi_identifier_from_length(filename, delimiter=delimiter)

    canvas = dcc.Graph(config={"modeBarButtonsToAdd": [
                        # "drawline",
                        # "drawopenpath",
                        "drawclosedpath",
                        # "drawcircle",
                        "drawrect",
                        "eraseshape"],
                        'toImageButtonOptions': {'format': 'png', 'filename': filename, 'scale': 1},
                            # disable scrollable zoom for now to control the scale bar
                        'edits': {'shapePosition': False}, 'scrollZoom': fullscreen_mode},
                        relayoutData={'autosize': True},
                        id=input_id,
                            style=style_canvas,
                        figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                               newshape = dict(line=dict(color="white")),
                                              xaxis=go.XAxis(showticklabels=False),
                                              yaxis=go.YAxis(showticklabels=False))})

    return dash_draggable.GridLayout(id='draggable', children=[canvas]) if draggable else canvas

def wrap_canvas_in_loading_screen_for_large_images(image=None, size_threshold=3000, hovertext=False, enable_zoom=False,
                                                   wrap=True, filename: str="canvas", delimiter: str="+++"):
    """
    Wrap the annotation canvas in a dcc.Loading screen if the dimensions of the image are larger than the threshold
    or
    if hovertext is used (slows down the canvas considerably)
    """
    # conditions for wrapping the canvas
    large_image = image is not None and (image.shape[0] > size_threshold or image.shape[1] > size_threshold)
    if (large_image or hovertext) and wrap:
        return dcc.Loading(render_default_annotation_canvas(fullscreen_mode=enable_zoom, filename=filename,
                        delimiter=delimiter), type="default", fullscreen=False, color=SessionTheme().widget_colour)
    else:
        return render_default_annotation_canvas(fullscreen_mode=enable_zoom, filename=filename, delimiter=delimiter)

def add_scale_value_to_figure(figure, image_shape, scale_value=None, font_size=12, x_axis_left=0.05, pixel_ratio=1,
                              invert=False, proportion=0.1):
    """
    add a scalebar value to a canvas figure based on the dimensions of the current image
    """
    if scale_value is None:
        scale_val = int(float(proportion * image_shape[1]) * float(pixel_ratio))
    else:
        scale_val = scale_value
    scale_annot = str(scale_val) + "μm"
    scale_text = f'<span style="color: white">{scale_annot}</span><br>'
    figure = go.Figure(figure)
    half = float(proportion) / 2
    # the midpoint of the annotation is set by the middle of 0.05 and 0.125 and an xanchor of center`
    x = float((x_axis_left + half) if not invert else (x_axis_left - half))
    figure.add_annotation(text=scale_text, font={"size": font_size}, xref='paper',
                       yref='paper',
                       # set the placement of where the text goes relative to the scale bar
                       x=x,
                       xanchor='center',
                       y=0.06,
                       # yanchor='bottom',
                       showarrow=False)
    return figure


def get_additive_image_with_masking(currently_selected, data_selection, canvas_layers, mask_config,
                                    mask_toggle, mask_selection, show_canvas_legend,
                                    mask_blending_level, add_mask_boundary, legend_text, annotation_size=12,
                                    proportion=0.1):
    """
    Generate an additiive image from one or more channel arrays. Optionally, project a mask on top of the additive image
    using a specified blend ratio with cv2
    """
    try:
        image = sum([np.asarray(canvas_layers[data_selection][elem]).astype(np.float32) for \
                 elem in currently_selected if \
                 elem in canvas_layers[data_selection].keys()]).astype(np.float32)
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
                          x0=x_axis_placement, y0=0.05, x1=(x_axis_placement + proportion),
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
    except (KeyError, AttributeError):
        return dash.no_update


def add_local_file_dialog(use_local_dialog=False, input_id="local-dialog-file"):
    if use_local_dialog:
        return dbc.Button(children=html.Span([html.I(className="fa-regular fa-folder-open",
        style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
        html.Div("Browse/read local files")], style={"display": "flex"}),
        id=input_id, className="mb-3", color=None, n_clicks=0, style={"margin-top": "10px"})
    else:
        return html.Div(id=input_id)

def invert_annotations_figure(cur_canvas: go.Figure):
    """
    Invert the annotations (scalebar and legend) on a canvas figure
    """
    if 'layout' in cur_canvas and 'annotations' in cur_canvas['layout']:
        cur_annotations = deepcopy(cur_canvas['layout']['annotations'])
    else:
        cur_annotations = []
    if 'layout' in cur_canvas and 'shapes' in cur_canvas['layout']:
        cur_shapes = deepcopy(cur_canvas['layout']['shapes'])
    else:
        cur_shapes = []
    for shape in cur_shapes:
        try:
            if 'y0' in shape and shape['y0'] == 0.05 and 'y1' in shape and shape['y1'] == 0.05:
                shape['x0'] = 1 - shape['x0']
                shape['x1'] = 1 - shape['x1']
        except IndexError:
            pass
    for annot in cur_annotations:
        try:
            if annot['y'] in [0.05, 0.06]:
                annot['x'] = 1 - annot['x']
        except IndexError:
            pass
    cur_canvas['layout']['annotations'] = cur_annotations
    cur_canvas['layout']['shapes'] = cur_shapes
    return cur_canvas

def set_range_slider_tick_markers(max_value, num_ticks=4):
    """
    Set the number and spacing of the tick markers used for the pixel range slider using the histogram maximum
    Note: the slider minimum is always set to 0
    """
    if float(max_value) <= 1.0:
        return dict([(i, str(i)) for i in [0, 1]]), float(round((float(max_value) / 10), 2))
    else:
        # set the default number of tick marks to 4
        # if the maximum value is less than 3, reduce the number of ticks accordingly
        if int(max_value) < 3:
            num_ticks = int(max_value) + 1
        # sets the dictionary for the string and int values to be shown in the pixel intensity range slider
        return dict([(int(i), str(int(i))) for i in list(np.linspace(0, int(max_value), num_ticks))]), 1


def generate_canvas_legend_text(blend_colour_dict, channel_order, aliases, legend_orientation="vertical",
                                use_cluster_annotations=False, cluster_colour_dict: dict=None,
                                data_selection: str=None):
    """
    Generate the string annotation text for a canvas based on the channels and selected colour of the channel
    """
    legend_text = ''
    gap = "" if legend_orientation == "vertical" else " "
    line_break = "<br>" if legend_orientation == "vertical" else ""
    # use only unique aliases in the legend to allow merging of identical channels
    aliases_used = []
    if not use_cluster_annotations:
        for image in channel_order:
        # if blend_colour_dict[image]['color'] not in ['#ffffff', '#FFFFFF']:
            label = aliases[image] if aliases is not None and image in aliases.keys() else image
            if label not in aliases_used:
                legend_text = legend_text + f'<span style="color:' \
                                    f'{blend_colour_dict[image]["color"]}"' \
                                    f'>{label}{gap}</span>{line_break}'
                aliases_used.append(label)
    elif use_cluster_annotations and cluster_colour_dict:
        try:
            # these will automatically be unique
            for clust in list(cluster_colour_dict[data_selection].keys()):
                legend_text = legend_text + f'<span style="color:' \
                                        f'{cluster_colour_dict[data_selection][clust]}"' \
                                        f'>{clust}{gap}</span>{line_break}'
        except KeyError:
            pass
    return legend_text


def set_x_axis_placement_of_scalebar(image_x_shape, invert_annot=False):
    """
    Set the x-axis placement of the scalebar using a formula based on the image width
    `image_x_shape`: The dimension, in pixels, of the x-axis (width) of the image in the canvas
    """
    x_axis_placement = 0.000025 * image_x_shape
    # make sure the placement is min 0.05 and max 0.1
    x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.15 else 0.05
    if invert_annot:
        x_axis_placement = 1 - x_axis_placement
    return x_axis_placement

def set_roi_identifier_from_length(dataset_selection, length_threshold=5, delimiter: str="+++"):
    """
    Set the output name for a dataset based on the length of the ROI name
    If the ROI name is below a certain length, output the entire dataset identifier
    """
    try:
        exp, slide, roi = split_string_at_pattern(dataset_selection, delimiter)
        # set a length limit: if the roi name is long enough to be informative, set as the output
        roi_name_use = roi if len(roi) > length_threshold else dataset_selection
        return roi_name_use
    except (KeyError, IndexError, ValueError):
        return dataset_selection

def update_canvas_filename(canvas_config: dict, roi_name: str=None, delimiter: str="+++"):
    """
    update the canvas config with the latest ROI name
    """
    if canvas_config and roi_name:
        try:
            canvas_config['toImageButtonOptions']['filename'] = str(set_roi_identifier_from_length(roi_name,
                                                                                    delimiter=delimiter))
            return canvas_config
        except KeyError:
            pass
    return canvas_config
