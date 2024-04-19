import os
import shutil
from dash.exceptions import PreventUpdate

def remove_ccramic_caches(directory):
    """
    Remove any ccramic caches from the specified directory
    """
    if os.access(directory, os.R_OK):
        # TODO: establish cleaning the tmp dir for any sub directory that has ccramic cache in it
        subdirs = [x[0] for x in os.walk(directory) if 'ccramic_cache' in x[0]]
        # remove any parent directory that has a ccramic cache in it
        for dir in subdirs:
            if os.access(os.path.dirname(dir), os.R_OK) and os.access(dir, os.R_OK):
                shutil.rmtree(os.path.dirname(dir))

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
