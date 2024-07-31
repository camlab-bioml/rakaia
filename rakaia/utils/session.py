import os
from typing import Union
import time
import shutil
from dash.exceptions import PreventUpdate

def remove_rakaia_caches(directory):
    """
    Remove any rakaia caches from the specified directory
    """
    if os.access(directory, os.R_OK):
        sub_dirs_found = [x[0] for x in os.walk(directory) if 'rakaia_cache' in x[0]]
        # remove any parent directory that has a rakaia cache in it
        for dir_found in sub_dirs_found:
            if os.access(os.path.dirname(dir_found), os.R_OK) and os.access(dir_found, os.R_OK):
                shutil.rmtree(os.path.dirname(dir_found), ignore_errors=True)

def non_truthy_to_prevent_update(input_obj):
    """
    Convert a non-truthy object to a raise PreventUpdate from dash
    """
    if not input_obj:
        raise PreventUpdate
    return input_obj

def validate_session_upload_config(cur_session_config: dict=None):
    """
    Validate the file upload session configuration hash table
    If there are no current uploads, or the bash doesn't exist, create it
    """
    return cur_session_config if cur_session_config is not None and 'uploads' in cur_session_config and \
            len(cur_session_config['uploads']) > 0 else {'uploads': []}

def channel_dropdown_selection(channels: dict=None, channel_names: dict=None):
    """
    Generate the list of channel dropdown options for a particular session
    Each element of the list is a dictionary with a label and value for the channel
    names are the editable display name for the channel, and values are the internal keys that do not change
    """
    if channels and channel_names:
        return [{'label': channel_names[i], 'value': i} for i in channels.keys() if len(i) > 0 and \
                i not in ['', ' ', None] and i in channel_names.keys()]
    return []

def sleep_on_small_roi(roi_dimensions: tuple=None, dim_threshold: Union[int, float]=400,
                       seconds_pause: Union[int, float]=2):
    """
    Initiate a sleep pause if an ROI has dimensions below the threshold. Allows for the canvas to render
    properly on ROI changes between ROIs of varying sizes
    """
    if roi_dimensions and all(dim <= dim_threshold for dim in roi_dimensions):
        time.sleep(seconds_pause)
