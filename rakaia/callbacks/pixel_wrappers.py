from typing import Union
from pathlib import Path
import dash
import os
from dash import html
import plotly.graph_objs as go
from rakaia.utils.alert import AlertMessage

GLOBAL_FILTER_KEYS = ["global_apply_filter", "global_filter_type", "global_filter_val", "global_filter_sigma"]

def generate_annotation_list(canvas_layout: Union[go.Figure, dict],
                             bulk_annot: bool=False):
    """
    Generate a preliminary annotation list to populate the annotation dict
    Keep separate from the annotation dict as new layouts will produce new region or shapes to annotate
    Each key is the unique tuple of elements in the annotation, and the value is the type required for parsing
    If bulk annot is enabled, then every shape is parsed. By default, only the most recent shape is added
    (it is assumed that the previous shapes have already been added).
    """
    # use the data collection as the highest key then use the canvas coordinates to uniquely identify a region
    # IMP: convert the dictionary to a sorted tuple to use as a key
    # https://stackoverflow.com/questions/1600591/using-a-python-dictionary-as-a-key-non-nested
    annotation_list = {}
    # Option 1: if zoom is used
    if isinstance(canvas_layout, dict) and 'shapes' not in canvas_layout:
        annotation_list[tuple(sorted(canvas_layout.items()))] = "zoom"
    # Option 2: if a shape is drawn on the canvas
    elif 'shapes' in canvas_layout and isinstance(canvas_layout, dict) and len(canvas_layout['shapes']) > 0:
        # only get the shapes that are a rect or path, the others are canvas annotations
        # Set which shapes to use based on the checklist either all or the most recent
        shapes_use = canvas_layout['shapes'] if bulk_annot else [canvas_layout['shapes'][-1]]
        for shape in shapes_use:
            if shape['type'] == 'path':
                annotation_list[shape['path']] = 'path'
            elif shape['type'] == "rect":
                key = {k: shape[k] for k in ('x0', 'x1', 'y0', 'y1')}
                annotation_list[tuple(sorted(key.items()))] = "rect"
    return annotation_list

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


def parse_local_path_imports(path: str, session_config: dict, error_config: dict):
    """
    Parse the path of either a local filepath or directory to retrieve the imaging datasets associated with it
    Return a dictionary error and a no update callback if there if the path does not correspond to a proper filepath
    or a directory
    """
    if os.path.isfile(path):
        session_config['uploads'].append(path)
        error_config["error"] = None
        return session_config, dash.no_update
    elif os.path.isdir(path):
        extensions = ["*.tiff", "*.mcd", "*.tif", "*.txt", "*.h5"]
        for extension in extensions:
            session_config['uploads'].extend(Path(path).glob(extension))
        session_config['uploads'] = [str(elem) for elem in session_config['uploads']]
        return session_config, dash.no_update
    error_config["error"] = AlertMessage().warnings["invalid_path"]
    return dash.no_update, error_config

def mask_options_from_json(config: dict):
    mask_options = ["mask_toggle", "mask_level", "mask_boundary", "mask_hover"]
    if 'mask' in config and all([key in config['mask'].keys() for key in mask_options]):
        return [val for val in config['mask'].values()]
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

def bounds_text(x_low: Union[int, float], x_high: Union[int, float], y_low: Union[int, float],
                y_high: Union[int, float]):
    """
    Generate the bounds text for the html preview Div
    """
    if None not in (x_low, x_high, y_low, y_high):
        return [html.Br(), html.H6(f"Current bounds: \n X: ({round(x_low, 2)}, {round(x_high, 2)}), "
                        f"Y: ({round(y_low, 2)}, {round(y_high, 2)})", style={"color": "black", "white-space": "pre"}),
     html.Br()], {"x_low": x_low, "x_high": x_high, "y_low": y_low, "y_high": y_high}
    return [], {}

def no_json_db_updates(error_config: dict=None):
    """
    Return a tuple of `dash.no_update` and error config objects when the JSON configuration or database
    configuration doesn't occur
    """
    return dash.no_update, dash.no_update, error_config, dash.no_update, dash.no_update, dash.no_update, \
        dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update, \
        dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
