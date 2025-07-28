"""Module containing functions and classes for setting variable session configurations
such as hover text, tab text, themes, download directory, etc.
"""

import os
import json
from pathlib import Path
from typing import Union
import h5py
from pydantic import BaseModel
import pandas as pd
import numpy as np
from dash_extensions.enrich import Serverside
from rakaia.utils.pixel import path_to_mask

class SessionTheme(BaseModel):
    """
    Sets the default theme elements for the session in the `widget_colour` attribute

    :return: None
    """
    widget_colour: str = "#0f4d92"

class TabText(BaseModel):
    """
    Holds the html-compatible text explanations for different tabs

    :return: None
    """
    dataset_preview: str = "ROIs that have been successfully imported are listed below. " \
                           "Selecting a row will load the ROI into the main canvas and channel gallery."
    panel: str = "Panel metadata consists of a list of biomarkers corresponding to one or more " \
                    "experiments. rakaia requires internal channel identifiers (stored under" \
                    " the channel name) that are used within rakaia sessions to identify individual biomarkers." \
                    " Channel labels may be edited under the final column of the metadata table; these " \
                    "labels will be applied to the canvas and session inputs."
    channel_tiles: str = "Each region is comprised of one or more images corresponding to the " \
                         "expression of a biomarker. Individual biomarker images, termed tiles, are visible in the " \
                         "channel gallery when an ROI is selected, and one or more biomarkers can be added to the canvas " \
                         "blend."
    region_gallery: str = "Generate a thumbnail for one or more ROIs contained in the current session with either the " \
                          "current blend parameters or a saved blend. These thumbnails may be generated " \
                          "randomly using query parameters, or from subsetting the UMAP plot under the quantification tab. " \
                          "Each thumbnail enables the specific ROI to be loaded into the main canvas."
    metadata: str = "Custom metadata in CSV format can be imported for one or multiple ROIs. Associations between " \
                    "pairs of metadata variables can be visualized below with categorical overlays."


class SessionServerside(Serverside):
    """
    Defines the string identification for Serverside objects depending on the session invocation.
    `use_unique_key` should be set to True for local runs where there are no concurrent users, and the user wishes to
    have the callbacks overwrite previous callback stores
    For public or sessions with concurrent users (i.e. Docker), `use_unique_key` should be set to False so that
    each callback invocation produces a unique Serverside cache

    :param data: The pickle-compatible data object to be stored
    :param key: The string key used to identify the pickled objects written to disk
    :param use_unique_key: Whether to use a unique key for every invocation. Should be `True` except for shared instances.
    :return: None
    """
    def __init__(self, data, key, use_unique_key: bool=True):
        self.use_unique_key = use_unique_key
        self.identifier = key
        key = key if self.use_unique_key else None
        super().__init__(value=data, key=key)

def create_download_dir(dest_dir):
    """
    Creates the download directory
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

class JSONSessionDocument:
    """
    Represents a JSON saved configuration of session and ROI variables
    Saved configurations include blend parameters for all channels, naming/aliases,
    global filters, and cluster colour annotations, if imported
    Used for local export to JSON or insertion into a mongoDB collection

    :param save_type: Whether the document will be saved as `json` (default) or to the mongoDB database (use `db`).
    :param user: Username
    :param document_name: User set name of the blend config
    :param blend_dict: Dictionary of current channel blend parameters
    :param selected_channel_list: List of channels in the current blend
    :param global_apply_filter: Whether a global filter has been applied
    :param global_filter_type: String specifying a global gaussian or median blur
    :param global_filter_val: Kernel size for the global gaussian or median blur
    :param global_filter_sigma: If global gaussian blur is applied, set the sigma value
    :param data_selection: String representation of the current session ROI selection
    :param cluster_assignments: Dictionary of mask cluster categories matching to a cluster color
    :param aliases: Dictionary matching channel keys to their displayed labels
    :param gating_dict: Dictionary of current gating parameters
    :param mask_toggle: Whether to show the mask over the channel blend
    :param mask_level: Set opacity of mask relative to the blend image. Takes a value between 0 and 100
    :param mask_boundary: Whether to include the object boundaries in the mask projection
    :mask_hover: Whether to include the mask object ID in the hover template
    """
    def __init__(self, save_type="json", user: str=None, document_name: str=None,
                 blend_dict: dict=None, selected_channel_list: list=None,
                 global_apply_filter: Union[list, bool]=False, global_filter_type: str="median",
                 global_filter_val: int = 3, global_filter_sigma: float=1.0,
                 data_selection: str=None, cluster_assignments: dict=None, aliases: dict=None,
                 gating_dict: dict=None, mask_toggle: bool=False, mask_level: Union[int, float]=35,
                               mask_boundary: bool=True, mask_hover: Union[bool, list]=False):
        if save_type not in ["json", "db"]:
            raise TypeError("The `save_type` provided should be one of: `json`, for local exports,"
                            "or `db`, for formatting a document for the mongoDB database")
        self.format = save_type
        self.document = {}
        if self.format == "db":
            self.document['user'] = user
            self.document['name'] = document_name
        if aliases is not None:
            for key in blend_dict.keys():
                if key in aliases.keys():
                    blend_dict[key]['alias'] = aliases[key]
        self.document['channels'] = blend_dict
        cluster_assignments = cluster_assignments[data_selection] if None not in \
                            (cluster_assignments, data_selection) and data_selection in \
                            cluster_assignments.keys() else None
        self.document['config'] = {"blend": selected_channel_list, "filter": {"global_apply_filter": global_apply_filter,
                                "global_filter_type": global_filter_type, "global_filter_val": global_filter_val,
                                "global_filter_sigma": global_filter_sigma}}
        self.document['cluster'] = cluster_assignments
        self.document['gating'] = gating_dict
        self.document['mask'] = {"mask_toggle": mask_toggle, "mask_level": mask_level,
                               "mask_boundary": mask_boundary, "mask_hover": mask_hover}
    def get_document(self) -> dict:
        """
        Return the JSON-style document

        :return: dictionary of session blend parameters in JSON/dict format.
        """
        return self.document

def panel_match(current_blend: Union[dict, None], new_blend: Union[dict, None]):
    """
    Check that a new imported panel from db or JSON matches the panel from a currently loaded ROI
    """
    try:
        current_blend = current_blend if isinstance(current_blend, list) else list(current_blend.keys())
        return None not in (current_blend, new_blend) and all(
            elem in new_blend['channels'] for elem in current_blend) and len(new_blend['channels']) == len(current_blend)
    except (KeyError, TypeError):
        return False

def all_roi_match(current_blend: Union[dict, None], new_blend: Union[dict, None], image_dict: Union[dict, None],
                  delimiter: str="+++"):
    """
    Cheek if the current session panel is compatible and matches with a panel attempting to be imported.
    The match occurs if all channel keys match as well as the panel length
    """
    try:
        return current_blend is None and new_blend is not None and all(len(image_dict[roi]) == \
                        len(new_blend['channels']) for roi in image_dict.keys() if delimiter in roi)
    except (KeyError, TypeError):
        return False

def write_blend_config_to_json(dest_dir, blend_dict, blend_layer_list, global_apply_filter,
                               global_filter_type, global_filter_val, global_filter_sigma,
                               data_selection: str=None, cluster_assignments: dict=None, aliases: dict=None,
                               gating_dict: dict=None, mask_toggle: bool=False, mask_level: Union[int, float]=35,
                               mask_boundary: bool=True, mask_hover: Union[bool, list]=False):
    """
    Write the session blend configuration dictionary to a JSON file
    """
    # write the aliases to the blend_dict if they exist
    param_json_path = str(os.path.join(dest_dir, 'param.json'))
    with open(param_json_path, "w") as outfile:
        dict_write = JSONSessionDocument("json", None, None, blend_dict, blend_layer_list, global_apply_filter,
                                         global_filter_type, global_filter_val, global_filter_sigma,
                                         data_selection, cluster_assignments, aliases, gating_dict,
                                         mask_toggle, mask_level, mask_boundary, mask_hover).get_document()
        json.dump(dict_write, outfile)
    return param_json_path

def write_canvas_shapes_to_json(dest_dir: Union[Path, str],
                                canvas_layout: Union[dict, None]=None):
    """
    Write the current canvas shapes to JSON
    """
    if canvas_layout is not None and 'shapes' in canvas_layout and canvas_layout['shapes']:
        param_json_path = str(os.path.join(dest_dir, 'canvas_shapes.json'))
        with open(param_json_path, "w") as outfile:
            json.dump(canvas_layout, outfile)
        return param_json_path
    return None

def write_session_data_to_h5py(dest_dir, metadata_frame, data_dict, data_selection, blend_dict, mask=None):
    """
    Write the current data dictionary and blend configuration to an h5py file
    """
    # TODO: add the global filter and blend list to the h5py output
    relative_filename = os.path.join(dest_dir, 'data.h5')
    h5_out = None
    try:
        h5_out = h5py.File(relative_filename, 'w')
    except OSError: os.remove(relative_filename)
    if h5_out is None:
        h5_out = h5py.File(relative_filename, 'w')

    meta_to_write = pd.DataFrame(metadata_frame) if metadata_frame is not None else \
        pd.DataFrame(data_dict['metadata'])
    for col in meta_to_write:
        meta_to_write[col] = meta_to_write[col].astype(str)
    h5_out.create_dataset('metadata', data=meta_to_write.to_numpy())
    h5_out.create_dataset('metadata_columns', data=meta_to_write.columns.values.astype('S'))
    h5_out.create_group(data_selection)
    for key, value in data_dict[data_selection].items():
        if key not in h5_out[data_selection]:
            h5_out[data_selection].create_group(key)
            if 'image' not in h5_out[data_selection][key] and value is not None:
                # use the mask if provided
                if mask is not None:
                    value[~mask] = 0
                h5_out[data_selection][key].create_dataset('image', data=value)
                if blend_dict is not None and key in blend_dict.keys():
                    for blend_key, blend_val in blend_dict[key].items():
                        data_write = str(blend_val) if blend_val is not None else "None"
                        h5_out[data_selection][key].create_dataset(blend_key, data=data_write)
                else:
                    pass
    try:
        h5_out.close()
    except (Exception,): pass
    return str(relative_filename)


def subset_mask_for_data_export(canvas_layout, array_shape):
    """
    Generate a numpy array mask from the last svg path shape in the canvas layout
    """
    mask = None
    try:
        for shape in canvas_layout['shapes']:
            if shape['type'] == 'path':
                path = shape['path']
                if mask is None:
                    mask = path_to_mask(path, array_shape)
                else:
                    new_mask = path_to_mask(path, array_shape)
                    mask = np.logical_or(mask, new_mask)
    except KeyError:
        pass
    return mask


def sort_channel_dropdown(channel_list: Union[dict, None]=None, sort_channels: bool=False):
    """
    Sort the channel dropdown component alphanumerically. By default, the dropdown menu lists channels
    in the order that they are contained in the origin files.
    """
    try:
        if sort_channels:
            channels_return = dict(sorted(channel_list.items(), key=lambda x: x[1].lower()))
        else:
            channels_return = channel_list
        return channels_return
    except AttributeError:
        return channel_list
