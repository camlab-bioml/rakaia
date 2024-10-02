import math
from io import BytesIO
import base64
from typing import Union
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
from PIL import Image
import numpy as np
from rakaia.utils.pixel import (
    resize_for_canvas,
    get_default_channel_upper_bound_by_percentile,
    apply_preset_to_array)

def replace_child_card_label(child: dict, aliases: dict):
    """
    Replace a `dbc.Card` title for a channel label with an updated channel alias from the session hash
    """
    if isinstance(child, dict) and 'children' in child and 'className' in child and \
            child['className'] == 'card-text' and 'id' in child and child['id'] in aliases.keys():
        child['children'] = aliases[child['id']]
    return child

def replace_child_hover_template(child: dict, aliases: dict):
    """
    Replace a `dbc.Hovertip` title for a channel thumbnail with an updated channel alias from the session hash
    """
    if isinstance(child, dict) and 'children' in child and 'target' in child and \
            'index' in child['target'] and child['target']['index'] in aliases.keys():
        child['children'] = f"Add {aliases[child['target']['index']]} to canvas"
    return child

def replace_channel_gallery_aliases(child, aliases_dict):
    """
    Recursively traverse the existing channel gallery and change the channel labels based
    on the alias dictionary
    """
    # if a channel bold label is found, update the alias
    child = replace_child_card_label(child, aliases_dict)
    # if a channel hover tip is found, update the hover text with the alias
    child = replace_child_hover_template(child, aliases_dict)
    if isinstance(child, dict):
        for value in child.values():
            replace_channel_gallery_aliases(value, aliases_dict)
    elif isinstance(child, list):
        for value in child:
            replace_channel_gallery_aliases(value, aliases_dict)
    return child
#
def gallery_image_identifiers(gallery_children: list):
    """
    Parse the current children for image labels
    """
    labels = []
    for child in gallery_children:
        if isinstance(child, dict):
            for sub_child in child['props']['children']['props']['children']:
                if sub_child['props']['children']:
                    for inner_child in sub_child['props']['children']:
                        if isinstance(inner_child['props'], dict) and 'id' in inner_child['props'] and \
                                'index' in inner_child['props']['id']:
                            labels.append(inner_child['props']['id']['index'])
    return labels

def gallery_image_src(gallery_children: list):
    """
    Parse the current gallery children for image src attributes
    """
    src = []
    for child in gallery_children:
        if isinstance(child, dict):
            for sub_child in child['props']['children']['props']['children']:
                if 'src' in sub_child['props']:
                    src.append(sub_child['props']['src'])
    return src

def channel_tiles_from_gallery(gallery_children: list):
    """
    Create a dictionary of tiles for HTML export from an existing gallery consisting of dbc children
    """
    return {key: {'label': key, 'tile': value} for key, value in
        zip(gallery_image_identifiers(gallery_children),
            gallery_image_src(gallery_children))}

def set_gallery_thumbnail_from_signal_retention(original_image: np.array, down_sampled_image: np.array,
                                                alternate_image: np.array,
                                                signal_ratio: Union[int, float],
                                                resize_signal_retention_threshold: Union[int, float] = 0.75,
                                                resize_dimension_threshold: int = 3000):
    """
    Set a thumbnail for a channel image based on the dimensions and signal retained from the down-sampled resize
    If the signal lost if sufficiently high and the image is below a certain size, return the original image. Otherwise,
    use the down-sampled image
    Dimension threshold is to prevent very large images from being used as thumbnails
    """
    return down_sampled_image if (signal_ratio > resize_signal_retention_threshold or
                                 any(size > resize_dimension_threshold for size in
                                     original_image.shape)) else alternate_image


def set_channel_thumbnail(canvas_layout: Union[dict, go.Figure], channel_image: Union[np.array, np.ndarray],
                          zoom_keys: list=None, toggle_gallery_zoom: bool=False):
    """
    Generate the numpy array for the greyscale channel thumbnail used for the preview gallery. Incorporates
    the option to use the canvas graph zoom to show just a subset region
    Return tuple: thumbnail image (subset by zoom or resized), and the alternate image based on the signal retention,
    used if the signal retained isn't sufficient
    """
    alternative = channel_image
    if zoom_keys and all(elem in canvas_layout for elem in zoom_keys) and toggle_gallery_zoom:
        x_range_low = math.floor(int(canvas_layout['xaxis.range[0]']))
        x_range_high = math.floor(int(canvas_layout['xaxis.range[1]']))
        y_range_low = math.floor(int(canvas_layout['yaxis.range[1]']))
        y_range_high = math.floor(int(canvas_layout['yaxis.range[0]']))
        try:
            if not x_range_high >= x_range_low or not y_range_high >= y_range_low:
                raise AssertionError
            image_render = channel_image[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                                        range(int(x_range_low), int(x_range_high), 1))]
            alternative = image_render
        except IndexError:
            image_render = channel_image
    else:
        image_render = resize_for_canvas(channel_image)
    return image_render, alternative

def verify_channel_tile(image_render: Union[np.array, np.ndarray], key: str,
                        raw_channel_array: Union[np.array, np.ndarray],
                        blend_colour_dict: dict, preset_dict: dict=None, preset_selection: str=None,
                        nclicks_preset: int=0,
                        toggle_scaling_gallery=False,
                        resize_signal_retention_threshold: float = 0.75,
                        resize_dimension_threshold: int = 3000,
                        single_channel_identifier: str=None):
    """
    Verify a channel tile. Checks the appropriate default scaling bounds and evaluates
    if a preset should be used or if the entire array should be used for signal retention
    """
    channel_key = single_channel_identifier if single_channel_identifier else key
    if toggle_scaling_gallery:
        try:
            if blend_colour_dict[key]['x_lower_bound'] is None:
                blend_colour_dict[key]['x_lower_bound'] = 0
            if blend_colour_dict[key]['x_upper_bound'] is None:
                blend_colour_dict[key]['x_upper_bound'] = \
                    get_default_channel_upper_bound_by_percentile(raw_channel_array)
            image_render = apply_preset_to_array(image_render,
                                                 blend_colour_dict[channel_key])
        except (KeyError, TypeError):
            pass
    if None not in (preset_selection, preset_dict) and nclicks_preset > 0:
        image_render = apply_preset_to_array(image_render, preset_dict[preset_selection])

    ratio = float(np.mean(image_render) / np.mean(raw_channel_array))
    # use the down-sampled image if the single retention is high enough, or
    # if the image is large (large images take longer to render in the DOM)
    try:
        image_render = set_gallery_thumbnail_from_signal_retention(raw_channel_array, image_render,
                        apply_preset_to_array(raw_channel_array,
                        blend_colour_dict[channel_key]).astype(np.uint8), ratio,
                        resize_signal_retention_threshold, resize_dimension_threshold)
    except KeyError as e:
        print(e)
        pass
    return image_render

def channel_tiles(gallery_dict, canvas_layout, zoom_keys, blend_colour_dict,
                                           preset_selection, preset_dict, aliases, nclicks_preset,
                                           toggle_gallery_zoom=False, toggle_scaling_gallery=False,
                                           resize_signal_retention_threshold: float = 0.75,
                                           resize_dimension_threshold: int = 3000,
                                           single_channel_identifier: str=None):
    """
    Generate the children for the image gallery comprised of the single channel images for one ROI
    """
    tiles = {}
    if gallery_dict is not None and len(gallery_dict) > 0:
        for key, value in gallery_dict.items():
            image_render, value = set_channel_thumbnail(canvas_layout, value, zoom_keys, toggle_gallery_zoom)
            image_render = verify_channel_tile(image_render, key, value, blend_colour_dict,
                                               preset_dict, preset_selection, nclicks_preset,
                                               toggle_scaling_gallery,
                                               resize_signal_retention_threshold,
                                               resize_dimension_threshold,
                                               single_channel_identifier)
            label = aliases[key] if aliases is not None and key in aliases.keys() else key
            tiles[key] = {"label": label, "tile": image_render}
    return tiles

# IMP: specifying n_clicks on button addition can trigger an erroneous selection
# https://github.com/facultyai/dash-bootstrap-components/issues/1047
def channel_tile_gallery_children(tiles: Union[dict, None]):
    """
    Generate the children for the image gallery comprised of the single channel images for one ROI
    """
    row_children = []
    for key, value in tiles.items():
            label = value['label'] if 'label' in value else key
            tile_image = value['tile']
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody([html.B(label,
                        className="card-text", id=key),
                        dbc.Button(children=html.Span(
                        [html.I(className="fa-solid fa-plus-circle",
                        style={"display": "inline-block", "margin-right": "7.5px",
                                "margin-top": "-5px"})],
                        style={"display": "flex", "float": "center", "justifyContent": "center"}),
                        id={'type': 'gallery-channel', 'index': key},
                        outline=False, color="light", className="me-1", size="m",
                        style={"padding": "5px", "margin-left": "10px",
                        "margin-top": "2.5px"}),
                        dbc.Tooltip(f'Add {label} to canvas',
                        target={'type': 'gallery-channel', 'index': key})]),
                        dbc.CardImg(src=Image.fromarray(tile_image).convert('RGB'),
                        bottom=True)]), width=3))
    return row_children


def roi_query_gallery_children(image_dict, col_width=4, max_size=28, max_aspect_ratio_tall=0.9):
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
                style = {"height": f"{max_size}rem", "width": f"{max_size * aspect_ratio}rem",
                         "justifyContent": "center"}
            else:
                style = None
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody([html.B(label, className="card-text"),
                    dbc.Button("Load",
                    id={'type': 'data-query-gallery', 'index': key},
                    outline=True, color="dark", className="me-1", size="sm",
                    style={"padding": "5px", "margin-left": "10px", "margin-top": "2.5px"})]),
                    dbc.CardImg(src=Image.fromarray(value.astype(np.uint8)),
                    bottom=True, style=style,
                    className='align-self-center')]),
                    width=col_width))
            roi_list.append(key)
    return row_children, roi_list


def channel_column_template(channel_name: str, channel_array: Union[np.array, np.ndarray, str]):
    """
    Set the HTML column template for a channel array to export in HTML format
    """
    chan_str = channel_array
    data_dir = ""
    if not isinstance(channel_array, str):
        channel_array = Image.fromarray(resize_for_canvas(channel_array)).convert('RGB')
        buffered = BytesIO()
        channel_array.save(buffered, format="png")
        base64_channel = base64.b64encode(buffered.getvalue())
        # base64_channel = base64.b64encode(channel_array)
        chan_str = base64_channel.decode()
        data_dir = "data:image/png;base64, "
    return f"<div class='column'>" \
            f"<h3 id={channel_name} style='margin: 15px;' class='card-text'>{channel_name} </b>" \
            f"<img style='max-width: 100%; max-height: 100%;' src='{data_dir}{str(chan_str)}'>" \
            "</div>"

def gallery_export_template(dest_file: str, tiles: dict, by_roi: bool=False,
                            num_cols: int=4):
    """
    Set the template for exporting the channel gallery in HTML format
    """
    cols_added = ""
    if by_roi:
        new_tiles = {}
        for key, value in tiles.items():
            new_tiles[key] = {'label': key, 'tile': value}
        tiles = new_tiles
    for value in tiles.values():
        cols_added = cols_added + channel_column_template(value['label'], value['tile'])

    col_settings = ":root {--gridCol: " + str(num_cols) + "; font-family: Arial;}"
    row_settings = ".row {display: grid; grid-template-columns: repeat(" + \
                    str(num_cols) + ", 1fr); gap: 10px;}"

    html_template = "<!DOCTYPE html> " \
                    "<html>" \
                    "<head>" \
                    "<style>" \
                    f"{col_settings}" \
                    f"{row_settings}" \
                    "h3 {max-width: 100%; overflow-wrap: anywhere;}" \
                    "</style>" \
                    "</head>" \
                    "<body>" \
                    "<div class='row'>" \
                    f"{cols_added}" \
                    "</div>" \
                    "</body>" \
                    "</html>"

    html_write = open(dest_file, "w")
    # Adding input data to the HTML file
    html_write.write(html_template)
    # Saving the data into the HTML file
    html_write.close()
    return dest_file
