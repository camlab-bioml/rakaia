"""Advanced functions that support the callbacks associated with
pixel-level operations (blended images)"""
import json
from typing import Union
from pathlib import Path
import dash
import os
from dash import html
import plotly.graph_objs as go
import pandas as pd
from natsort import natsorted, ns
from setuptools.command.alias import alias

from rakaia.io.session import SessionServerside
from rakaia.parsers.object import get_quantification_filepaths_from_drag_and_drop, parse_masks_from_filenames
from rakaia.utils.alert import AlertMessage

GLOBAL_FILTER_KEYS = ["global_apply_filter", "global_filter_type", "global_filter_val", "global_filter_sigma"]

class AnnotationList:
    """
    Generate a preliminary annotation list to populate the annotation dict
    Keep separate from the annotation dict as new layouts will produce new region or shapes to annotate
    Each key is the unique tuple of elements in the annotation, and the value is the type required for parsing
    If bulk annot is enabled, then every shape is parsed. By default, only the most recent shape is added
    (it is assumed that the previous shapes have already been added).

    :param canvas_layout: Dictionary corresponding to the layout of the current canvas, containing coordinates or shapes.
    :param bulk_annot: Whether to use all annotations or just the most recent

    :return: None
    """
    def __init__(self, canvas_layout: Union[go.Figure, dict],
                             bulk_annot: bool=False):
        self.canvas_layout = canvas_layout
        self.bulk_annot = bulk_annot
        self.annotations = {}
        self.check_annotation_for_zoom(self.canvas_layout)
        self.check_annotation_for_shapes(self.canvas_layout)

    def check_annotation_for_zoom(self, canvas_layout: Union[go.Figure, dict]):
        """
        Check if the current layout corresponds to a zoom event

        :param canvas_layout: Dictionary corresponding to the layout of the current canvas, containing coordinates or shapes.

        :return: None
        """
        if isinstance(canvas_layout, dict) and 'shapes' not in canvas_layout:
            self.annotations[tuple(sorted(canvas_layout.items()))] = "zoom"

    def check_annotation_for_shapes(self, canvas_layout: Union[go.Figure, dict]):
        """
        Check if the current layout contains any shapes.

        :param canvas_layout: Dictionary corresponding to the layout of the current canvas, containing coordinates or shapes.

        :return: None
        """
        if 'shapes' in canvas_layout and isinstance(canvas_layout, dict) and len(canvas_layout['shapes']) > 0:
            # only get the shapes that are a rect or path, the others are canvas annotations
            # Set which shapes to use based on the checklist either all or the most recent
            shapes_use = canvas_layout['shapes'] if self.bulk_annot else [canvas_layout['shapes'][-1]]
            for shape in shapes_use:
                if shape['type'] == 'path':
                    self.annotations[shape['path']] = 'path'
                elif shape['type'] == "rect":
                    key = {k: shape[k] for k in ('x0', 'x1', 'y0', 'y1')}
                    self.annotations[tuple(sorted(key.items()))] = "rect"

    def get_annotations(self):
        """
        Return the list of annotations for a single annotation event.

        :return: Dictionary of annotations with keys corresponding to the annotation, and values describing the type
        """
        return self.annotations

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
        extensions = ["*.tiff", "*.mcd", "*.tif", "*.txt", "*.h5", "*.h5ad"]
        for extension in extensions:
            session_config['uploads'].extend(Path(path).glob(extension))
        session_config['uploads'] = [str(elem) for elem in session_config['uploads']]
        return session_config, dash.no_update
    error_config["error"] = AlertMessage().warnings["invalid_path"]
    return dash.no_update, error_config


class SteinbockParserKeys:
    """
    Define the subdirectories and permissible file extensions for parsing a steinbock output directory
    """
    sub_directories = ['quantification', 'mcd', 'deepcell']
    extensions = ['.tiff', '.tif', '.h5ad', '.mcd', '.txt']
    base_names = ['umap_coordinates']

def is_steinbock_dir(directory):
    """
    Check if a local filepath is a directory for steinbock outputs
    """
    if directory and os.path.isdir(directory):
        sub_dirs = os.listdir(directory)
        return all(elem in sub_dirs for elem in SteinbockParserKeys.sub_directories)
    return False

def parse_steinbock_subdir(sub_dir, single_file_return: bool=False):
    """
    Parse a specified steinbock output subdirectory
    """
    files_found = []
    if os.path.isdir(sub_dir):
        for root_dir, sub_sirs, sub_files in os.walk(os.path.abspath(sub_dir)):
            for search in sub_files:
                found_file = os.path.join(root_dir, search)
                basename, file_extension = os.path.splitext(found_file)
                if file_extension in SteinbockParserKeys.extensions or \
                        any(name in basename for name in SteinbockParserKeys.base_names):
                    files_found.append(found_file)
    if files_found and single_file_return:
        return files_found[0]
    return files_found if files_found else None

def recursive_parse_umap_coordinates(sub_dir: Union[str, Path]):
    """
    Parse a steinbock project output directory recursively for a list of CSV files that contain
    UMAP coordinate lists (i.e. those generated from the steinbock snakemake pipeline)
    """
    return natsorted([str(i) for i in Path(sub_dir).rglob('*coordinates.csv')], alg=ns.REAL)

def check_valid_upload(upload: Union[dict, list]):
    """
    Check for a valid upload component (existing filenames successfully parsed)
    """
    if upload is not None and 'uploads' in upload:
        return upload if upload['uploads'] else dash.no_update
    return upload if upload else dash.no_update

def parse_steinbock_dir(directory, error_config,
                        send_message: bool=False, **kwargs):
    """
    Parse a steinbock output directory. Returns a list of mcd/raw image files, list of mask names,
    quantification/.h5ad filepaths, UMAP coordinate dataframe, and scaling JSON dictionary
    """
    error_config = {"error": 'Error'} if not error_config else error_config
    mcd_files = parse_steinbock_subdir(os.path.join(directory, 'mcd'))
    mask_files = parse_steinbock_subdir(os.path.join(directory, 'deepcell', 'cell'))
    export_files = parse_steinbock_subdir(os.path.join(directory, 'export'))
    umap_files = recursive_parse_umap_coordinates(os.path.join(directory, 'export'))
    quant = [str(file) for file in export_files if file.endswith('.h5ad')] if export_files else []
    umap = umap_files[0] if umap_files else []
    umap_return = SessionServerside(pd.read_csv(umap, names=['UMAP1', 'UMAP2'],
                header=0).to_dict(orient="records"), **kwargs) if umap else dash.no_update
    scaling = parse_steinbock_scaling(directory) if parse_steinbock_scaling(directory) is not None else dash.no_update
    message = 'Successfully parsed' if mcd_files else 'Error parsing'
    error_config['error'] = f'{message} steinbock output directory {str(directory)}'
    # decide if we want to notify if successful parsing
    error_config = error_config if send_message else dash.no_update
    return check_valid_upload({'uploads': mcd_files, 'from_steinbock': True}), \
        error_config, parse_masks_from_filenames(None, mask_files), \
        get_quantification_filepaths_from_drag_and_drop(None, quant), umap_return, scaling

def parse_steinbock_umap(directory: Union[str, Path],
                         seek_png: bool=True):
    """
    Return a list of UMAP coordinate plots from the `steinbock` pipeline.
    Use seek_png to search for the UMAP plots (True), or set False to retrieve the CSV coordinate lists
    """
    if is_steinbock_dir(directory):
        search_term = "*.png" if seek_png else "*coordinates.csv"
        return natsorted([str(i) for i in Path(directory).rglob(search_term)], alg=ns.REAL)
    return []

def parse_steinbock_scaling(directory: Union[str, Path]):
    """
    Parse a steinbock pipeline output directory for a `scaling.json` file used for channel scaling visualization
    """
    files = [str(i) for i in Path(directory).rglob("*scaling.json")]
    json_parsed = json.load(open(files[0])) if files else None
    return json_parsed['channels'] if (json_parsed and 'channels' in json_parsed and
                                       json_parsed['channels']) else None

def umap_coordinates_from_gallery_click(umap_selection: str,
                                        directory: Union[str, Path],
                                        use_unique_key: bool=True,
                                        session_id: str=""):
    """
    Return a `pd.DataFrame` of UMAP coordinates based on a UMAP gallery selection by partial filename
    """
    if umap_selection and directory:
        umap_files = parse_steinbock_umap(directory, seek_png=False)
        umap_return = None
        for umap_dist in umap_files:
            if str(Path(umap_dist).stem) == f'{str(umap_selection)}_coordinates':
                umap_return = SessionServerside(pd.read_csv(str(umap_dist), names=['UMAP1', 'UMAP2'],
                header=0).to_dict(orient="records"),
                key=f"umap_coordinates_{session_id}", use_unique_key=use_unique_key)
                break
        return umap_return if umap_return is not None else dash.no_update
    return dash.no_update

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
        return [html.Br(), html.H6(f"Current bounds: \n X: ({round(x_low, 1)}, {round(x_high, 1)}), "
                        f"Y: ({round(y_low, 1)}, {round(y_high, 1)})",
                            style={"color": "black", "white-space": "pre", "width": "95%", "max-width": "95%"}),
     html.Br()], {"x_low": x_low, "x_high": x_high, "y_low": y_low, "y_high": y_high}
    return [], {}

def pixel_values_text(x_coord: Union[float, int], y_coord: Union[float, int],
                      image_dict: dict, data_selection: str, channels_selected: list,
                      mask_dict: dict, mask_selection: str, channel_aliases: Union[dict, None]=None):
    """
    Generate the test for pixel values (raw channel + mask ID) from a click event
    """
    children = [html.Span(f"x: {x_coord}, y: {y_coord}\n"), html.Br()]
    for channel in channels_selected:
        chan_represent = str(channel_aliases[channel]) if (channel_aliases
                        and channel in channel_aliases.keys()) else channel
        children.append(html.Span(f" {chan_represent}: "
                        f"{str(image_dict[data_selection][channel][y_coord, x_coord])}\n"))
        children.append(html.Br())
    if mask_dict and mask_selection in mask_dict.keys():
        children.append(html.Span(f" Mask ID: "
        f"{str(mask_dict[mask_selection]['raw'][y_coord, x_coord])}\n"))
        children.append(html.Br())
    return children

def no_json_db_updates(error_config: dict=None):
    """
    Return a tuple of `dash.no_update` and error config objects when the JSON configuration or database
    configuration doesn't occur
    """
    return dash.no_update, dash.no_update, error_config, dash.no_update, dash.no_update, dash.no_update, \
        dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update, \
        dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def disable_gallery_by_roi(datasets: Union[list, None]=None):
    """
    Toggle if the channel gallery should be searchable by ROI depending on how many datasets are currently imported
    """
    return False if (datasets and isinstance(datasets, list) and len(datasets) > 1) else True
