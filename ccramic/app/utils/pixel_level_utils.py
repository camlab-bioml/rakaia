# from imctools.converters import ome2analysis
# from imctools.converters import ome2histocat
# from imctools.converters import mcdfolder2imcfolder
# from imctools.converters import exportacquisitioncsv
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageColor
import plotly.graph_objects as go
import plotly.express as px
from skimage import draw
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from dash.exceptions import PreventUpdate

def split_string_at_pattern(string, pattern="+++"):
    return string.split(pattern)
#
# def get_luma(rbg):
#     return 0.2126 * rbg[0] + 0.7152 * rbg[1] + 0.0722 * rbg[2]
#
#
# def generate_tiff_stack(tiff_dict, tiff_list, colour_dict):
#     # image = recolour_greyscale(tiff_dict[tiff_list[0]], colour_dict[tiff_list[0]])
#     # for other in tiff_list[1:]:
#     #     image = image + recolour_greyscale(tiff_dict[other], colour_dict[other]
#     return Image.fromarray(sum([recolour_greyscale(tiff_dict[elem], colour_dict[elem]) for elem in tiff_list]))


def recolour_greyscale(array, colour):
    if colour not in ['#ffffff', '#FFFFFF']:
        image = Image.fromarray(array.astype(np.uint8))
        image = image.convert('RGB')
        red, green, blue = ImageColor.getcolor(colour, "RGB")

        array = np.array(image)

        new_array = np.empty((array.shape[0], array.shape[1], 3))
        new_array[:, :, 0] = red
        new_array[:, :, 1] = green
        new_array[:, :, 2] = blue

        converted = new_array * (np.array(image) / 255)
        # print(converted)
        return converted.astype(np.uint8)

    else:
        image = Image.fromarray(array.astype(np.uint8))
        image = image.convert('RGB')
        return np.array(image).astype(np.uint8)

def get_area_statistics_from_rect(array, x_range_low, x_range_high, y_range_low, y_range_high):
    try:
        subset = array[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                          range(int(x_range_low), int(x_range_high), 1))]
        return np.average(subset), np.amax(subset), np.amin(subset)
    except IndexError:
        return None, None, None


def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)


def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def get_area_statistics_from_closed_path(array, svgpath):
    """
    Subset an array based on coordinates contained within a svg path drawn on the canvas
    """
    # https://dash.plotly.com/annotations?_gl=1*9dqxqk*_ga*ODM0NzUyNzQ3LjE2NjQyODUyNDc.*_ga_6G7EE0JNSC*MTY4MzU2MDY0My4xMDUuMS4xNjgzNTYyNDM3LjAuMC4w

    masked_array = path_to_mask(svgpath, array.shape)
    # masked_subset_data = ma.array(array, mask=masked_array)
    return np.average(array[masked_array]), np.amax(array[masked_array]), np.amin(array[masked_array])


def convert_to_below_255(array):
    return array if np.max(array) < 65000 else (array // 256).astype(np.uint8)


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
    except TypeError:
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
        array = array * scale_factor
    return array


def pixel_hist_from_array(array, subset_number=1000000):
    # try:
    # IMP: do not use the conversion to L as it will automatically set the max to 255
    # array = np.array(Image.fromarray(array.astype(np.uint8)).convert('L'))
    hist_data = np.hstack(array)
    max_hist = np.max(array)
    hist = np.random.choice(hist_data, subset_number) if hist_data.shape[0] > subset_number else hist_data
    # add the largest pixel to ensure that hottest pixel is included in the distribution
    try:
        hist = np.concatenate([np.array(hist), np.array([max_hist])])
    except ValueError:
        pass
    return go.Figure(px.histogram(hist, range_x=[min(hist), max(hist)]), layout_xaxis_range=[0, max(hist)]), \
        int(np.max(array))
    # except ValueError:
    #     print("error")
    #     return pixel_hist_from_array(np.array(Image.fromarray(array.astype(np.uint8)).convert('L')))


def apply_preset_to_array(array, preset):
    preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
    if isinstance(preset, dict) and all([elem in preset.keys() for elem in preset_keys]):
        array = filter_by_upper_and_lower_bound(array, preset['x_lower_bound'], preset['x_upper_bound'])
        if preset['filter_type'] == "median" and preset['filter_val'] is not None:
            array = median_filter(array, int(preset['filter_val']))
        elif preset['filter_val'] is not None:
            array = gaussian_filter(array, int(preset['filter_val']))
        return array


def apply_preset_to_blend_dict(blend_dict, preset_dict):
    """
    Populate the blend dict from a preset dict
    """
    assert all([key in blend_dict.keys() for key in preset_dict.keys()])
    for key, value in preset_dict.items():
        # do not change the color from a preset
        if key != "color":
            blend_dict[key] = value
    return blend_dict


def get_all_images_by_channel_name(upload_dict, channel_name):
    """
    Get all the images in a session dictionary from a channel name for the gallery view
    """
    images = {}
    for exp in list(upload_dict.keys()):
        if 'metadata' not in exp:
            for slide in upload_dict[exp].keys():
                for acq in upload_dict[exp][slide].keys():
                    for channel in upload_dict[exp][slide][acq].keys():
                        if channel == channel_name:
                            string = f"{exp}_{slide}_{acq}"
                            if upload_dict[exp][slide][acq][channel] is not None:
                                images[string] = upload_dict[exp][slide][acq][channel]
    return images


def validate_incoming_metadata_table(metadata, upload_dict):
    """
    Validate the incoming metadata sheet on custom upload against the data dictionary.
    The incoming metadata sheet must have the following characteristics:
        - be on the same length as every ROI in the dataset
        - have a column named "Column Label" that can be copied for editing
    """
    try:
        assert isinstance(metadata, pd.DataFrame)
        assert "Channel Label" in metadata.columns
        assert "Channel Name" in metadata.columns
        for exp in list(upload_dict.keys()):
            if 'metadata' not in exp:
                for slide in upload_dict[exp].keys():
                    for acq in upload_dict[exp][slide].keys():
                        # assert that for each ROI, the length is the same as the number of rows in the metadata
                        assert len(upload_dict[exp][slide][acq].keys()) == len(metadata.index)
        return metadata
    except (AssertionError, AttributeError):
        return None


def create_new_coord_bounds(window_dict, x_request, y_request):
    """
    Create a new window based on an xy coordinate request. The current zoom level is maintained
    and the requested coordinate is approximately the middle of the new window
    """
    try:
        assert all([value is not None for value in window_dict.values()])
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
    except (AssertionError, KeyError):
        return None

def copy_values_within_nested_dict(dict, current_data_selection, new_data_selection):
    """
    Copy the blend dictionary parameters (colour, filtering, scaling) from one acquisition/ROI in a nested
    dictionary to another
    """
    cur_exp, cur_slide, cur_acq = split_string_at_pattern(current_data_selection)
    new_exp, new_slide, new_acq = split_string_at_pattern(new_data_selection)

    if new_exp not in list(dict.keys()):
        dict[new_exp] = {}
    if new_slide not in list(dict[new_exp].keys()):
        dict[new_exp][new_slide] = {}
    if new_acq not in list(dict[new_exp][new_slide].keys()):
        dict[new_exp][new_slide][new_acq] = {}

    for key, value in dict[cur_exp][cur_slide][cur_acq].items():
        dict[new_exp][new_slide][new_acq][key] = value
    return dict

def per_channel_intensity_hovertext(channel_list):
    """
    generate custom hovertext for the annotation canvas that shows the individual pixel intensities of ll
    channels selected in the hover template. Assumes that the data has been added as customdata through
    np.stack((channels), axis=-1)
    """
    data_index = 0
    hover_template = "x: %{x}, y: %{y} <br>"
    try:
        assert isinstance(channel_list, list)
        for elem in channel_list:
            assert channel_list.index(elem) == data_index
            hover_template = hover_template + f"{str(elem)}: " + "%{customdata[" + f"{data_index}]" + "} <br>"
            data_index += 1
    except AssertionError:
        pass
    hover_template = hover_template + "<extra></extra>"
    return hover_template

def get_default_channel_upper_bound_by_percentile(array, percentile=99, subset_number=1000000):
    """
    Get a reasonable upper bound default on a channel with a percentile of the pixels
    """
    array_stack = np.hstack(array)
    data = np.random.choice(array_stack, subset_number) if array.shape[0] > subset_number else array_stack
    return float(np.percentile(data, percentile))

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
        channel_order = [list(aliases.keys())[list(aliases.values()).index(elem['Current canvas blend'])] for \
                         elem in rowdata]
        return channel_order
    else:
        raise PreventUpdate
