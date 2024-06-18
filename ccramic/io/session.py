import os
import json
import h5py
import pandas as pd
import numpy as np
from ccramic.utils.pixel_level_utils import path_to_mask
from dash_extensions.enrich import Serverside
from typing import Union
from pydantic import BaseModel

class SessionTheme(BaseModel):
    """
    Sets the default theme elements for the session
    """
    widget_colour: str = "#0f4d92"

class TabText(BaseModel):
    """
    Holds the html-compatible text explanations for different tabs
    """
    dataset_preview: str = "ROIs that have been successfully imported are listed below. " \
                           "Selecting a row will load the ROI into the main canvas and channel gallery."
    metadata: str = "Panel metadata consists of a list of biomarkers corresponding to one or more " \
                    "experiments. ccramic requires internal channel identifiers (stored under" \
                    " the channel name) that are used within ccramic sessions to identify individual biomarkers." \
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


class SessionServerside(Serverside):
    """
    This class defines the string identification for Serverside objects depending on the session invocation
    `use_unique_key` should be set to True for local runs where there are no concurrent users, and the user wishes to
    have the callbacks overwrite previous callback stores
    For public or sessions with concurrent users (i.e. Docker), `use_unique_key` should be set to False so that
    each callback invocation produces a unique Serverside cache
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
    def get_document(self):
        return self.document

def panel_length_match(current_blend: Union[dict, None], new_blend: Union[dict, None]):
    """
    Check that a new imported panel from db or JSON matches the length of the existing session
    """
    try:
        return None not in (current_blend, new_blend) and len(current_blend) == len(new_blend['channels'])
    except (KeyError, TypeError):
        return False

def all_roi_match(current_blend: Union[dict, None], new_blend: Union[dict, None], image_dict: Union[dict, None],
                  delimiter: str="+++"):
    try:
        return current_blend is None and new_blend is not None and all([len(image_dict[roi]) == \
                        len(new_blend['channels']) for roi in image_dict.keys() if delimiter in roi])
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

def write_session_data_to_h5py(dest_dir, metadata_frame, data_dict, data_selection, blend_dict, mask=None):
    """
    Write the current data dictionary and blend configuration to an h5py file
    """
    # TODO: add the global filter and blend list to the h5py output
    relative_filename = os.path.join(dest_dir, 'data.h5')
    hf = None
    try:
        hf = h5py.File(relative_filename, 'w')
    except OSError:
        os.remove(relative_filename)
    if hf is None:
        hf = h5py.File(relative_filename, 'w')

    meta_to_write = pd.DataFrame(metadata_frame) if metadata_frame is not None else \
        pd.DataFrame(data_dict['metadata'])
    for col in meta_to_write:
        meta_to_write[col] = meta_to_write[col].astype(str)
    hf.create_dataset('metadata', data=meta_to_write.to_numpy())
    hf.create_dataset('metadata_columns', data=meta_to_write.columns.values.astype('S'))
    hf.create_group(data_selection)
    for key, value in data_dict[data_selection].items():
        if key not in hf[data_selection]:
            hf[data_selection].create_group(key)
            if 'image' not in hf[data_selection][key] and value is not None:
                # use the mask if provided
                if mask is not None:
                    value[~mask] = 0
                hf[data_selection][key].create_dataset('image', data=value)
                if blend_dict is not None and key in blend_dict.keys():
                    for blend_key, blend_val in blend_dict[key].items():
                        data_write = str(blend_val) if blend_val is not None else "None"
                        hf[data_selection][key].create_dataset(blend_key, data=data_write)
                else:
                    pass
    try:
        hf.close()
    except (Exception,):
        pass

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
    try:
        if sort_channels:
            channels_return = dict(sorted(channel_list.items(), key=lambda x: x[1].lower()))
        else:
            channels_return = channel_list
        return channels_return
    except AttributeError:
        return channel_list
