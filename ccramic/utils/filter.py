import dash
from typing import Union
import numpy as np
import cv2
from scipy.ndimage import median_filter

def apply_filter_to_channel(channel_array: Union[np.array, np.ndarray]=None, filter_chosen: Union[bool, list]=True,
                                filter_name: str="median", filter_value: int=3, filter_sigma: Union[int, float]=1.0):
    """
    Apply a median or gaussian filter to a channel with constraints for the filter type
    """
    if filter_chosen and filter_name:
        if filter_name == "median" and int(filter_value) >= 1:
            try:
                channel_array = median_filter(channel_array, int(filter_value))
            except ValueError:
                pass
        else:
            # array = gaussian_filter(array, int(filter_value))
            if int(filter_value) % 2 != 0 and int(filter_value) >= 1:
                channel_array = cv2.GaussianBlur(channel_array, (int(filter_value),
                                int(filter_value)), float(filter_sigma))
    return channel_array


def return_current_channel_blend_params(blend_dict: dict, selected_channel: str=None):
    """
    Return the current blend parameters (colour and individual filter) for a specific channel
    """
    try:
        filter_type = blend_dict[selected_channel]['filter_type']
        filter_val = blend_dict[selected_channel]['filter_val']
        filter_sigma = blend_dict[selected_channel]['filter_sigma']
        color = blend_dict[selected_channel]['color']
        return filter_type, filter_val, filter_sigma, color
    except KeyError:
        return None, None, None, None

def return_current_or_default_filter_apply(apply_channel_filter: Union[list, bool], filter_type: str="median",
                                           filter_val: int=3, filter_sigma: float=1.0):
    """
    Evaluate the state of the channel filter, and prevent an update, or return the value for the selected channel
    """
    if apply_channel_filter and None not in (filter_type, filter_val, filter_sigma):
        to_apply_filter = dash.no_update
    else:
        to_apply_filter = [' Apply/refresh filter'] if None not in (filter_type, filter_val, filter_sigma) else []
    return to_apply_filter

def return_current_or_default_filter_param(current_filter_param: Union[str, int, float]=None,
                                           new_filter_param: Union[str, int, float]=None):
    """
    Evaluate the state of the channel filter, and prevent an update, or return the value for the selected channel
    """
    if new_filter_param == current_filter_param:
        param_filter_return = dash.no_update
    else:
        param_filter_return = new_filter_param if new_filter_param is not None else current_filter_param
    return param_filter_return

def return_current_or_default_channel_color(current_color: Union[str, dict]=None,
                                           new_color: Union[str, dict]=None):
    """
    Evaluate the state of the channel color, and prevent an update, or return the value for the selected channel
    """
    if new_color == current_color['hex']:
        color_return = dash.no_update
    else:
        color_return = dict(hex=new_color) if new_color is not None and new_color not in \
                                          ['#FFFFFF', '#ffffff'] else dash.no_update
    return color_return


def return_current_default_params_with_preset(filter_type: str="median", filter_val: int=3, filter_sigma: float=1.0,
                                              color: Union[str, dict]=None):
    """
    Return the default blend parameters (color and filter) from the current values when using a preset
    """
    to_apply_filter = [' Apply/refresh filter'] if None not in (filter_type, filter_val) else []
    filter_type_return = filter_type if filter_type is not None else "median"
    filter_val_return = filter_val if filter_val is not None else 3
    filter_sigma_return = filter_sigma if filter_sigma is not None else 1.0
    color_return = dict(hex=color) if color is not None and color not in \
                                      ['#FFFFFF', '#ffffff'] else dash.no_update
    return to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return


def set_blend_parameters_for_channel(blend_dict: dict, channel_selection:str=None,
                                     filter_name: str=None, filter_value: Union[int, float]=3,
                                     filter_sigma: Union[int, float]=1.0, clear: bool=False):
    """
    Set the channel-specific filter values (type, either median or gaussian, filter kernel value, and filter sigma
    value) for a specific channel in the blend dictionary
    """
    if blend_dict and channel_selection and channel_selection in blend_dict:
        blend_dict[channel_selection]['filter_type'] = filter_name if not clear else None
        blend_dict[channel_selection]['filter_val'] = filter_value if not clear else None
        blend_dict[channel_selection]['filter_sigma'] = filter_sigma if not clear else None
    return blend_dict
