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
