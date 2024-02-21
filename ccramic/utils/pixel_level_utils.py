import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageColor
import plotly.graph_objects as go
import plotly.express as px
from skimage import draw
from scipy import ndimage
from scipy.ndimage import median_filter
from dash.exceptions import PreventUpdate
import cv2
import re
import random
import numexpr as ne
from typing import Union

def split_string_at_pattern(string, pattern="+++"):
    return string.split(pattern)


def set_array_storage_type_from_config(array_type="float"):
    """
    Set the array storage type within the application. Options are either 32-byte float or 16-byte int
    """
    if array_type not in ["float", "int"]:
        raise TypeError("the array type requested must be either float or int")
    return np.float32 if array_type == "float" else np.uint16

def is_rgb_color(value):
    """
    Return if a particular string is in the format of RGB hex code or not
    # https://stackoverflow.com/questions/20275524/how-to-check-if-a-string-is-an-rgb-hex-string
    """
    _rgbstring = re.compile(r'#[a-fA-F0-9]{6}$')
    return bool(_rgbstring.match(value))

def generate_default_swatches(config):
    """
    Return a list of default swatches for the mantine ColorPicker based on the contents of the config
    """
    DEFAULTS = ["#FF0000", "#00FF00", "#0000FF", "#00FAFF", "#FF00FF", "#FFFF00", "#FFFFFF"]
    try:
        # set the split based on if the swatches are given as a string or list
        if isinstance(config['swatches'], str):
            split = config['swatches'].split(',')
        elif isinstance(config['swatches'], list):
            split = config['swatches']
        else:
            split = None
        if split is not None:
            swatches = [str(i) for i in split if is_rgb_color(i)] if \
            config['swatches'] is not None else DEFAULTS
            swatches = swatches if len(swatches) > 0 else DEFAULTS
            return swatches
        else:
            return DEFAULTS
    except KeyError:
        return DEFAULTS


def recolour_greyscale(array, colour):
    """
    Convert a greyscale image into an RGB with a designated colour
    """
    if colour not in ['#ffffff', '#FFFFFF']:
        image = Image.fromarray(array)
        image = image.convert('RGB')
        red, green, blue = ImageColor.getcolor(colour, "RGB")

        array = np.array(image)

        new_array = np.empty((array.shape[0], array.shape[1], 3))
        new_array[:, :, 0] = red
        new_array[:, :, 1] = green
        new_array[:, :, 2] = blue

        converted = ne.evaluate("new_array * image / 255")
        return converted.astype(np.uint8)

    else:
        image = Image.fromarray(array)
        image = image.convert('RGB')
        return np.array(image)

def get_area_statistics_from_rect(array, x_range_low, x_range_high, y_range_low, y_range_high):
    """
    Return a series of region statistics for a rectangular slice of a channel array
    mean, max, min, and integrated (total) signal
    """
    try:
        subset = array[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                          range(int(x_range_low), int(x_range_high), 1))]
        return np.average(subset), np.max(subset), np.min(subset), np.sum(subset)
    except IndexError:
        return None, None, None

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    # trim the indices to only those that are inside of the border of the image
    # new_cols = np.array([])
    # new_rows = np.array([])
    col_inside = cc < shape[1]
    row_inside = rr < shape[0]
    both_inside = np.logical_and(col_inside, row_inside)
    rr = rr[both_inside]
    cc = cc[both_inside]
    # for row, col in zip(rr, cc):
    #     if row < shape[0] and col < shape[1]:
    #         new_cols = np.append(new_cols, [int(col)])
    #         new_rows = np.append(new_rows, [int(row)])
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def get_bounding_box_for_svgpath(svgpath):
    """
    Return the x and y min and max that defines the bounding box for a non-convex or convex svgpath
    Return values are: x_min, x_max, y_min, y_max
    """
    cols, rows = path_to_indices(svgpath).T
    return min(cols), max(cols), min(rows), max(rows)


def get_area_statistics_from_closed_path(array, svgpath):
    """
    Subset an array based on coordinates contained within a svg path drawn on the canvas
    Return a series of region statistics for a rectangular slice of a channel array
    mean, max, min, and integrated (total) signal
    """
    # https://dash.plotly.com/annotations?_gl=1*9dqxqk*_ga*ODM0NzUyNzQ3LjE2NjQyODUyNDc.*_ga_6G7EE0JNSC*MTY4MzU2MDY0My4xMDUuMS4xNjgzNTYyNDM3LjAuMC4w

    masked_array = path_to_mask(svgpath, array.shape)
    # masked_subset_data = ma.array(array, mask=masked_array)
    return np.average(array[masked_array]), np.max(array[masked_array]), np.min(array[masked_array]), \
        np.sum(array[masked_array])


# def convert_to_below_255(array):
#     return array if np.max(array) < 65000 else (array // 256).astype(np.uint8)


def resize_for_canvas(image, basewidth=400, return_array=True):
    image = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    if return_array:
        to_return = np.array(image.resize((basewidth, hsize), Image.Resampling.LANCZOS))
    else:
        to_return = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return to_return


def make_metadata_column_editable(column_name):
    # only allow the channel label column to be edited
    # return "Label" in column_name or column_name == "Channel Label"
    return column_name == "ccramic Label"


def filter_by_upper_and_lower_bound(array, lower_bound, upper_bound):
    """
    Filter an array by an upper and lower bound on the pixel values.
    Filter on the lower bound: removes any pixels less than the lower bound
    Filter on the upper bound: sets the upper bound as the new max intensity and scales all pixels
    relative to the new max.
    Uses linear scaling instead of 0 to max upper bound scaling: pixels close to the boundary of the lower bound
    are scaled relative to their intensity to the lower bound instead of the full scale factor
    """
    # if issparse(array):
    #     array = array.toarray(order='F')
    # https://github.com/BodenmillerGroup/histocat-web/blob/c598cd07506febf0b7c209626d4eb869761f2e62/backend/histocat/core/image.py
    # array = np.array(Image.fromarray(array).convert('L'))
    # original_max = np.max(array) if np.max(array) > 255 else 255
    lower_bound = float(lower_bound) if lower_bound is not None else None
    upper_bound = float(upper_bound) if upper_bound is not None else None
    if lower_bound is None:
        lower_bound = 0
    # array = np.where(array < lower_bound, 0, array)
    # try linear scaling from the lower bound to upper bound instead of 0 to upper
    # subtract the lower bound from all elements and retain those above 0
    # allows better gradual scaling around the lower bound threshold
    try:
        if upper_bound >= 0:
            array = np.where(array > upper_bound, upper_bound, array)
    except (TypeError, ValueError):
        pass
    array = np.where((array - lower_bound) > 0, (array - lower_bound), 0)
    if upper_bound is not None:
        try:
            scale_factor = 255 / (upper_bound - lower_bound)
        except ZeroDivisionError:
            scale_factor = 255
    else:
        scale_factor = 1
    if scale_factor > 0 and scale_factor != 1:
        array = ne.evaluate("array * scale_factor")
    return array


def pixel_hist_from_array(array, subset_number=1000000, keep_max=True):
    """
    Generate a pixel histogram from a channel array. If the number of array elements is larger than the subset number,
    create a down-sample
    If `keep_max` is True, then the final histogram will always retain the max of the original array cast
    to an integer so that it matches the range slider
    """
    # try:
    # IMP: do not use the conversion to L as it will automatically set the max to 255
    # array = np.array(Image.fromarray(array.astype(np.uint8)).convert('L'))
    # set the array cast type based on the max
    cast_type = np.uint16 if np.max(array) > 1 else np.float32
    hist_data = np.hstack(array).astype(cast_type)
    max_hist = float(np.max(array)) if int(np.max(array)) > 1 else 1
    hist = np.random.choice(hist_data, subset_number).astype(cast_type) if \
        hist_data.shape[0] > subset_number else hist_data
    # add the largest pixel to ensure that hottest pixel is included in the distribution
    # ensure that the min of the hist max is 1
    try:
        if keep_max:
            hist = np.concatenate([np.array(hist).astype(cast_type),
                                   np.array([max_hist]).astype(cast_type)])
    except ValueError:
        pass
    return go.Figure(px.histogram(hist, range_x=[min(hist), max_hist]), layout_xaxis_range=[0, max_hist]), \
        float(np.max(array))

def upper_bound_for_range_slider(array):
    """
    Return the pixel max of a channel array for the range slider, or return 1 if the max value is less than 1
    """
    return float(np.max(array)) if float(np.max(array)) > 1.0 else 1.0


def apply_preset_to_array(array, preset):
    preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
    if isinstance(preset, dict) and all([elem in preset.keys() for elem in preset_keys]):
        array = filter_by_upper_and_lower_bound(array, preset['x_lower_bound'], preset['x_upper_bound'])
        if preset['filter_type'] == "median" and preset['filter_val'] is not None and \
                int(preset['filter_val']) >= 1:
            try:
                array = median_filter(array, int(preset['filter_val']))
            except ValueError:
                pass
        elif preset['filter_val'] is not None:
            if int(preset['filter_val']) % 2 != 0 and int(preset['filter_val']) >= 1:
                array = cv2.GaussianBlur(array, (int(preset['filter_val']), int(preset['filter_val'])),
                                         int(preset['filter_sigma']))
        return array

def apply_preset_to_blend_dict(blend_dict, preset_dict):
    """
    Populate the blend dict from a preset dict
    """
    if not all([key in blend_dict.keys() for key in preset_dict.keys()]): raise AssertionError
    for key, value in preset_dict.items():
        # do not change the color from a preset
        if key != "color":
            blend_dict[key] = value
    return blend_dict


def get_all_images_by_channel_name(upload_dict, channel_name):
    """
    Get all the images in a session dictionary from a channel name for the gallery view
    """
    # TODO: modify this function to accommodate lazy loading
    images = {}
    for roi in list(upload_dict.keys()):
        if 'metadata' not in roi:
            for channel in upload_dict[roi].keys():
                if channel == channel_name:
                    if upload_dict[roi][channel] is not None:
                        images[roi] = upload_dict[roi][channel]
    return images


def validate_incoming_metadata_table(metadata, upload_dict):
    """
    Validate the incoming metadata sheet on custom upload against the data dictionary.
    The incoming metadata sheet must have the following characteristics:
        - be on the same length as every ROI in the dataset
        - have a column named "Column Label" that can be copied for editing
    """
    if isinstance(metadata, pd.DataFrame) and "Channel Label" in metadata.columns and upload_dict is not None and \
        all([len(upload_dict[roi]) == len(metadata.index) for roi in list(upload_dict.keys()) if \
             roi not in ['metadata', 'metadata_columns']]):
        return metadata
    return None


def create_new_coord_bounds(window_dict, x_request, y_request):
    """
    Create a new window based on an xy coordinate request. The current zoom level is maintained
    and the requested coordinate is approximately the middle of the new window
    """
    try:
        if not all([value is not None for value in window_dict.values()]): raise AssertionError
        # first cast the bounds as int, then cast as floats and add significant digits
        # 634.5215773809524
        x_request = float(x_request) + 0.000000000000
        y_request = float(y_request) + 0.000000000000
        x_low = float(min(float(window_dict['x_high']), float(window_dict['x_low'])))
        x_high = float(max(float(window_dict['x_high']), float(window_dict['x_low'])))
        y_low = float(min(float(window_dict['y_high']), float(window_dict['y_low'])))
        y_high = float(max(float(window_dict['y_high']), float(window_dict['y_low'])))
        midway_x = abs(float((x_high - x_low))) / 2
        midway_y = abs(float((y_high - y_low))) / 2
        new_x_low = float(float(x_request - midway_x) + 0.000000000000)
        new_x_high = float(float(x_request + midway_x) + 0.000000000000)
        new_y_low = float(float(y_request - midway_y) + 0.000000000000)
        new_y_high = float(float(y_request + midway_y) + 0.000000000000)
        return new_x_low, new_x_high, new_y_low, new_y_high
    except KeyError:
        return None

def per_channel_intensity_hovertext(channel_list):
    """
    generate custom hovertext for the annotation canvas that shows the individual pixel intensities of ll
    channels selected in the hover template. Assumes that the data has been added as customdata through
    np.stack((channels), axis=-1)
    """
    data_index = 0
    hover_template = "x: %{x}, y: %{y} <br>"
    if not isinstance(channel_list, list): return hover_template + "<extra></extra>"
    for elem in channel_list:
        if not channel_list.index(elem) == data_index: return hover_template + "<extra></extra>"
        hover_template = hover_template + f"{str(elem)}: " + "%{customdata[" + f"{data_index}]" + "} <br>"
        data_index += 1
    hover_template = hover_template + "<extra></extra>"
    return hover_template

def get_default_channel_upper_bound_by_percentile(array, percentile=99, subset_number=1000000):
    """
    Get a reasonable upper bound default on a channel with a percentile of the pixels
    """
    array_stack = np.hstack(array)
    data = np.random.choice(array_stack, subset_number) if array.shape[0] > subset_number else array_stack
    upper_percentile = float(np.percentile(data, percentile))
    return upper_percentile if upper_percentile > 0 else 1.0

def delete_dataset_option_from_list_interactively(remove_clicks, cur_data_selection, cur_options):
    """
    On button prompt, remove a dataset option from the options list.
    """
    if remove_clicks > 0 and None not in (cur_data_selection, cur_options):
        return_list = cur_options.copy()
        return_list.remove(cur_data_selection)
        return return_list, None, [], None
    else:
        raise PreventUpdate

def set_channel_list_order(set_order_clicks, rowdata, channel_order, current_blend, aliases, triggered_id):
    """
    Set the blend order of channels in the canvas based on either the existing order of addition,
    or the sorting from a dash-ag-grid that is passed as rowdata
    """
    channel_order = [] if channel_order is None or len(channel_order) < 1 else channel_order
    # input 1: if a channel is added or removed
    if triggered_id == "image_layers" and current_blend is not None and len(current_blend) > 0:
        for channel in current_blend:
            if channel not in channel_order:
                channel_order.append(channel)
        # make sure to remove any channels that are no longer selected while maintaining order
        return [elem for elem in channel_order if elem in current_blend]
    # option 2: if a unique order is set by the draggable grid
    elif triggered_id == "set-sort" and rowdata is not None and set_order_clicks > 0:
        # imp: when taking the order from the dash grid, these are the values, so need to convert back to keys
        channel_order = [list(aliases.keys())[list(aliases.values()).index(elem['Channel'])] for \
                         elem in rowdata]
        return channel_order
    else:
        return []


def select_random_colour_for_channel(blend_dict, current_channel, default_colours):
    """
    Loop through the default colours and select a colour in the sequence for the current channel
    if the colour is not used. Otherwise, do not update the blend dictionary
    """
    for default in default_colours:
        # check if any of the colours have been used yet. if not, select the next available
        # only updates a channel if the default is None or white (#FFFFFF), which implies the defaults
        if not any([channel['color'] == default for channel in blend_dict.values()]) and \
                blend_dict[current_channel]['color'] in ['#FFFFFF', None, '#ffffff']:
            blend_dict[current_channel]['color'] = default
            break
    return blend_dict


def random_hex_colour_generator(number=10):
    """
    Generate a list of random hex colours. The number provided will be the length of the list
    """
    #
    r = lambda: random.randint(0, 255)
    colours = []
    index = 0
    while index < number:
        colour = '#%02X%02X%02X' % (r(), r(), r())
        if colour not in colours:
            colours.append(colour)
            index += 1
    return colours


def get_additive_image(layer_dict: dict, channel_list: list) -> np.array:
    image_shape = layer_dict[channel_list[0]].shape
    image = np.zeros(image_shape)
    for elem in channel_list:
        blend = layer_dict[elem]
        image = ne.evaluate("image + blend")
    return image.astype(np.float32)

def get_first_image_from_roi_dictionary(roi_dictionary):
    """
    Return the first image in a dictionary that specifies an ROI. This assumes that
    all of the other channel arrays in the dictionary have the same shape
    """
    first_image_name = list(roi_dictionary.keys())[0]
    image_for_validation = roi_dictionary[first_image_name]
    return image_for_validation


def apply_filter_to_array(image, global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma):
    """
    Apply a filter to an array from a dictionary of global filter values
    Note: incorrect values applied to the array will not return an error, but will return the original array,
    as this function is meant to be used in the application
    """
    if global_filter_type not in ['gaussian', 'median']:
        raise TypeError("The global filter type should be either gaussian or median.")
    global_filter_applied = (isinstance(global_apply_filter, bool) and global_apply_filter) or (
        isinstance(global_apply_filter, list) and len(global_apply_filter) > 0)
    if global_filter_applied and None not in (global_filter_type, global_filter_val) and \
            int(global_filter_val) % 2 != 0:
        if global_filter_type == "median" and int(global_filter_val) >= 1:
            try:
                image = cv2.medianBlur(image, int(global_filter_val))
            except (ValueError, cv2.error):
                pass
        elif global_filter_type == "gaussian":
            # array = gaussian_filter(array, int(filter_value))
            if int(global_filter_val) >= 1:
                image = cv2.GaussianBlur(image, (int(global_filter_val),
                                                 int(global_filter_val)),
                                                 float(global_filter_sigma))
    return image


def no_filter_chosen(current_blend_dict: dict, channel: str, filter_chosen: Union[list, str]):
    """
    Evaluates whether the currently selected channel has no filter applied, and the session
    filter is set to None
    """
    return  current_blend_dict[channel]['filter_type'] is None and \
            current_blend_dict[channel]['filter_val'] is None and \
            current_blend_dict[channel]['filter_sigma'] is None and \
            len(filter_chosen) == 0
def channel_filter_matches(current_blend_dict: dict, channel: str, filter_chosen: Union[list, str],
                           filter_name: str="median", filter_value: int = 3, filter_sigma: Union[int, float] = 1.0):
    """
    Evaluates whether the current channel's filters match the filter parameters currently
    set in the session
    """
    return  current_blend_dict[channel]['filter_type'] == filter_name and \
            current_blend_dict[channel]['filter_val'] == filter_value and \
            current_blend_dict[channel]['filter_sigma'] == filter_sigma and \
            len(filter_chosen) > 0


def ag_grid_cell_styling_conditions(blend_dict: dict, current_blend: list, data_selection: str,
                                    channel_aliases: dict=None):
    """
    Generate the cell styling conditions for the dash ag grid that displays the current channels
    and their colours
    """
    cell_styling_conditions = []
    if blend_dict is not None and current_blend is not None and data_selection is not None:
        for key in current_blend:
            try:
                if key in blend_dict.keys() and blend_dict[key]['color'] != '#FFFFFF':
                    label = channel_aliases[key] if channel_aliases is not None and \
                                                    key in channel_aliases.keys() else key
                    cell_styling_conditions.append({"condition": f"params.value == '{label}'",
                                                    "style": {"color": f"{blend_dict[key]['color']}"}})
            except KeyError:
                pass
    return cell_styling_conditions
