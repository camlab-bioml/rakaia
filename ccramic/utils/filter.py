import dash
from typing import Union

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
