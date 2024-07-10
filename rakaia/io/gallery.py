import dash_bootstrap_components as dbc
from dash import html
from PIL import Image
import numpy as np
import math
from rakaia.utils.pixel import (
    resize_for_canvas,
    get_default_channel_upper_bound_by_percentile,
    apply_preset_to_array)

def generate_channel_tile_gallery_children(gallery_dict, canvas_layout, zoom_keys, blend_colour_dict,
                                           preset_selection, preset_dict, aliases, nclicks_preset,
                                           toggle_gallery_zoom=False, toggle_scaling_gallery=False):
    """
    Generate the children for the image gallery comprised of the single channel images for one ROI
    """
    row_children = []
    if gallery_dict is not None and len(gallery_dict) > 0:
        for key, value in gallery_dict.items():
            if all([elem in canvas_layout for elem in zoom_keys]) and toggle_gallery_zoom:
                x_range_low = math.floor(int(canvas_layout['xaxis.range[0]']))
                x_range_high = math.floor(int(canvas_layout['xaxis.range[1]']))
                y_range_low = math.floor(int(canvas_layout['yaxis.range[1]']))
                y_range_high = math.floor(int(canvas_layout['yaxis.range[0]']))
                try:
                    if not x_range_high >= x_range_low: raise AssertionError
                    if not y_range_high >= y_range_low: raise AssertionError
                    image_render = value[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                                            range(int(x_range_low), int(x_range_high), 1))]
                except IndexError:
                    image_render = value
            else:
                image_render = resize_for_canvas(value)
            if toggle_scaling_gallery:
                try:
                    if blend_colour_dict[key]['x_lower_bound'] is None:
                        blend_colour_dict[key]['x_lower_bound'] = 0
                    if blend_colour_dict[key]['x_upper_bound'] is None:
                        blend_colour_dict[key]['x_upper_bound'] = \
                                get_default_channel_upper_bound_by_percentile(
                            value)
                    image_render = apply_preset_to_array(image_render,
                                                     blend_colour_dict[key])
                except (KeyError, TypeError):
                    pass
            if None not in (preset_selection, preset_dict) and nclicks_preset > 0:
                image_render = apply_preset_to_array(image_render, preset_dict[preset_selection])

            label = aliases[key] if aliases is not None and key in aliases.keys() else key
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody([html.B(label, className="card-text"),
                                                                dbc.Button(children=html.Span(
                                                                    [html.I(className="fa-solid fa-plus-circle",
                                                                            style={"display": "inline-block",
                                                                                   "margin-right": "7.5px",
                                                                                   "margin-top": "-5px"})],
                                                                    style={"display": "flex", "float": "center",
                                                                           "justifyContent": "center"}),
                                                                    id={'type': 'gallery-channel',
                                                                        'index': key},
                                                                    outline=False, color="light",
                                                                    className="me-1", size="m",
                                                                    style={"padding": "5px",
                                                                           "margin-left": "10px",
                                                                           "margin-top": "2.5px"}),
                                dbc.Tooltip(f'Add {label} to canvas', target={'type': 'gallery-channel', 'index': key}),
                                                                ]),
                                              dbc.CardImg(src=Image.fromarray(image_render).convert('RGB'),
                                                          bottom=True)]), width=3))
    return row_children

def generate_roi_query_gallery_children(image_dict, col_width=4, max_size=28, max_aspect_ratio_tall=0.9):
    """
    Return a series of columns and rows for across dataset ROi queries. Each element will be a dbc Card preview
    of the ROI image from the current blend dictionary with the option to click to load into the canvas
    Additionally returns a list of the queried ROI names to avoid overlap with additional queries
    """
    row_children = []
    roi_list = []
    if image_dict is not None and len(image_dict) > 0:
        for key, value in image_dict.items():
            # add the dimensions to the label as a list to provide a line break
            label = f"{key}: ({value.shape[1]}x{value.shape[0]})"
            aspect_ratio = int(value.shape[1]) / int(value.shape[0])
            # implement a cap on very tall images to avoid a lot of white space
            if aspect_ratio < max_aspect_ratio_tall:
                style = {"height": f"{max_size}rem", "width": f"{max_size * aspect_ratio}rem", "justifyContent": "center"}
            else:
                style = None
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody([html.B(label, className="card-text"),
                                                            dbc.Button("Load",
                                                                                  id={'type': 'data-query-gallery',
                                                                                      'index': key},
                                                                                  outline=True, color="dark",
                                                                                  className="me-1", size="sm",
                                                                       style={"padding": "5px",
                                                                              "margin-left": "10px",
                                                                              "margin-top": "2.5px"})]),
                                              dbc.CardImg(src=Image.fromarray(value.astype(np.uint8)),
                                                          bottom=True, style=style, className='align-self-center')]),
                                    width=col_width))
            roi_list.append(key)
    return row_children, roi_list
