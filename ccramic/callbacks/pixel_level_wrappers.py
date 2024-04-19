import dash
import os

from dash.exceptions import PreventUpdate

from ccramic.utils.alert import AlertMessage
from pathlib import Path

GLOBAL_FILTER_KEYS = ["global_apply_filter", "global_filter_type", "global_filter_val", "global_filter_sigma"]

def parse_global_filter_values_from_json(config_dict):
    """
    parse the global filter values from a config JSON
    """
    global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma = \
        dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if 'filter' in config_dict and all([elem in config_dict['filter'].keys() for elem in GLOBAL_FILTER_KEYS]):
        global_apply_filter = config_dict['filter']['global_apply_filter']
        global_filter_type = config_dict['filter']['global_filter_type']
        global_filter_val = config_dict['filter']['global_filter_val']
        global_filter_sigma = config_dict['filter']['global_filter_sigma']
    return global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma


def parse_local_path_imports(path: str, import_type: str, session_config: dict, error_config: dict):
    """
    Parse the path of either a local filepath or directory to retrieve the imaging datasets associated with it
    Return a dictionary error and a no update callback if there if the path does not correspond to a proper filepath
    or a directory
    """
    if import_type == "filepath":
        if os.path.isfile(path):
            session_config['uploads'].append(path)
            error_config["error"] = None
            return session_config, dash.no_update
        else:
            error_config["error"] = AlertMessage().warnings["invalid_filepath"]
            return dash.no_update, error_config
    elif import_type == "directory":
        if os.path.isdir(path):
            # valid_files = []
            extensions = ["*.tiff", "*.mcd", "*.tif", "*.txt", "*.h5"]
            for extension in extensions:
                session_config['uploads'].extend(Path(path).glob(extension))
            session_config['uploads'] = [str(elem) for elem in session_config['uploads']]
            return session_config, dash.no_update
        else:
            error_config["error"] = AlertMessage().warnings["invalid_directory"]
            return dash.no_update, error_config
    raise PreventUpdate
