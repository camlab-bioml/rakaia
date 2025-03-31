"""Module containing utility functions for marker quantification and object gating sets
"""

from typing import Union
import numpy as np
import pandas as pd
from dash import html
from steinbock.measurement.intensities import (
    IntensityAggregation,
    measure_intensites)
from steinbock.measurement.regionprops import measure_regionprops
from rakaia.utils.object import validate_mask_shape_matches_image
from rakaia.utils.pixel import split_string_at_pattern, high_low_values_from_zoom_layout


class DistributionTableColumns:
    """
    Define the basic columns for the quantification distribution table generated with UMAP interactivity in the
    `columns` attribute
    :return: None
    """
    columns = ["Value", "Counts", "Proportion"]

def mask_object_counter_preview(mask_dict: dict=None, mask_selection: str=None):
    """
    Generate a string preview of the number of objects in a current mask, used in the quantification modal
    Returns a string that is compatible as a html.B child, or an empty list of the required inputs do not exist
    """
    if mask_dict and mask_selection and mask_selection in mask_dict:
        try:
            ids = np.unique(mask_dict[mask_selection]['raw'])
            return f"{len(ids[ids != 0])} mask objects"
        except KeyError:
            return []
    return []


def quantify_one_channel(image, mask):
    """
    Takes an array of one channel with the matching mask and creates a vector of the mean values per
    segmented object
    """
    # reshape the image to retain only the first two dimensions
    image = image.reshape((image.shape[0], image.shape[1]))
    mask = mask.reshape((mask.shape[0], mask.shape[1]))
    if validate_mask_shape_matches_image(mask, image):
        # Get unique list of cell IDs. Remove '0' which corresponds to background
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids != 0]
        # IMP: the cell ids start at 1, but the array positions start at 0, so offset the array positions by 1
        offset = 1 if min(cell_ids) == 1 else 0
        # Output expression matrix
        expr_mat = np.zeros((1, len(cell_ids)))
        for cell in cell_ids:
            is_cell = mask == cell
            # IMP: only apply the offset in the positions, not for the actual cell id
            expr_mat[:, int(cell - offset)] = np.mean(image[is_cell])
        return expr_mat
    return None

def quantify_multiple_channels_per_roi(channel_dict, mask, data_selection, channels_to_quantify, aliases=None,
                                       dataset_options=None, delimiter: str="+++", mask_name: str=None):
    """
    Quantify multiple channels for a single ROI and concatenate into a dataframe with cells
    as rows and channels + metadata as columns
    """
    try:
        exp, slide, roi_name = split_string_at_pattern(data_selection, pattern=delimiter)
        # trim quantification channels to ones with an existing array, for example if quantifying spatial such as xenium
        channels_to_quantify = [chan for chan in channels_to_quantify if
                                chan in channel_dict[data_selection].keys() and
                                channel_dict[data_selection][chan] is not None]
        array = np.stack(
            [channel_dict[data_selection][channel] for channel in list(channel_dict[data_selection].keys()) if
             channel in channels_to_quantify], axis=0)
        chan_names = []
        for chan in channels_to_quantify:
            if aliases and chan in aliases.keys():
                chan_names.append(aliases[chan])
            else:
                chan_names.append(chan)
        channel_frame = measure_intensites(array, mask, chan_names, IntensityAggregation.MEAN).dropna()
        description_name = mask_name
        sample_name = mask_name
        if dataset_options is not None:
            for dataset in dataset_options:
                exp, slide, roi = split_string_at_pattern(dataset, pattern=delimiter)
                if roi == "acq" and roi == roi_name:
                    # this condition is used if non-mcd files are being quantified
                    description_name = mask_name
                    sample_name = mask_name

        channel_frame['description'] = description_name
        # channel_frame['cell_id'] = pd.Series(range(0, (int(np.max(mask)))), dtype='int64')
        channel_frame['cell_id'] = [int(i) for i in channel_frame.index]
        channel_frame['sample'] = sample_name
        props = ['area', 'centroid', 'axis_major_length', 'axis_minor_length', 'eccentricity']
        try:
            region_props = measure_regionprops(array, mask, props)
            to_return = channel_frame.join(region_props).reset_index(drop=True)
        except TypeError: to_return = channel_frame
        return to_return
    except ValueError:
        return None

def concat_quantification_frames_multi_roi(existing_frame, new_frame, new_data_selection, delimiter: str="+++"):
    """
    Concatenate a quantification frame from one ROI to an existing frame that may contain quantification
    results for one or more ROIs.
    If the columns do not match or the ROI has already been quantified,
    then the new frame will replace the existing frame
    """
    empty_master_frame = existing_frame is None or existing_frame.empty
    empty_new_frame = new_frame is None or new_frame.empty
    exp, slide, roi_name = split_string_at_pattern(new_data_selection, pattern=delimiter)
    if not empty_master_frame and not empty_new_frame:
        if roi_name not in existing_frame['sample'].tolist() and roi_name not in \
        existing_frame['description'].tolist() and set(list(existing_frame.columns)) == set(list(new_frame.columns)):
            return pd.concat([existing_frame, new_frame], axis=0, ignore_index=True)
        return new_frame
    if empty_master_frame and not empty_new_frame:
        return new_frame
    return existing_frame


def populate_gating_dict_with_default_values(current_gate_dict: dict=None, parameter_list: list=None):
    """
    Populate a gating dict with default normalized values of 0 to 1 if the
    parameter doesn't yet exist
    """
    current_gate_dict = {} if current_gate_dict is None else current_gate_dict
    for param in parameter_list:
        if param not in current_gate_dict:
            current_gate_dict[param] = {'lower_bound': 0.0, 'upper_bound': 1.0}
    return current_gate_dict

def update_gating_dict_with_slider_values(current_gate_dict: dict=None, gate_selected: str=None,
                                          gating_vals: tuple=(0.0, 1.0)):
    """
    Update the current gating dictionary with the range slider values for a specific parameter from
    the gating modifier dropdown
    """
    current_gate_dict = {gate_selected: {}} if current_gate_dict is None else current_gate_dict
    if gate_selected and gate_selected not in current_gate_dict:
        current_gate_dict[gate_selected] = {'lower_bound': None, 'upper_bound': None}
    current_gate_dict[gate_selected]['lower_bound'] = float(min(gating_vals))
    current_gate_dict[gate_selected]['upper_bound'] = float(max(gating_vals))
    return current_gate_dict

class GatingZoomSubset:
    """
    Subset a list of gated object ids to a current canvas zoom

    :param mask: Current ROI mask selection with segmented objects as integers
    :param canvas_layout: Dictionary containing the current `relayoutData` for the main canvas
    :param show_gating: Whether the gating zoom subset should be computed or not. (Default: True)

    :return: None
    """
    def __init__(self, mask: Union[np.array, None],
                 canvas_layout: Union[dict, None],
                 show_gating: bool=True):
        self.mask = mask
        self.layout = canvas_layout
        self.show_gating = show_gating

    def _gating_objects_in_zoom(self, obj_list: Union[list, None]=None):
        """
        Generate the list of gated object ids in the current view

        :param obj_list: List of total gated objects in the current ROI

        :return: list of objects (Subset of `obj_list` contained in the current zoom level)
        """
        if None not in (obj_list, self.layout) and self.mask is not None and \
                all(elem in self.layout for elem in ['xaxis.range[0]', 'xaxis.range[1]',
                'yaxis.range[0]', 'yaxis.range[1]']) and self.show_gating:
            x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(self.layout, cast_type=int)
            mask_subset = self.mask[y_low:y_high, x_low:x_high]
            return list(set(list(np.unique(mask_subset))).intersection(set(obj_list)))
        return None

    def generate_list(self, obj_list: Union[list, None]):
        """
        Generate the list of gated object ids in the current view

        :param obj_list: List of total gated objects in the current ROI

        :return: list of objects (Subset of `obj_list` contained in the current zoom level)
        """
        return self._gating_objects_in_zoom(obj_list)

def gating_label_children(use_gating: bool = True, gating_dict: dict = None,
                          current_gating_params: list=None, object_id_list: Union[list, None]=None,
                          from_custom_list: bool=False, zoom_obj_list: Union[list, None]=None):
    """
    Generate the HTML legend for the current parameters used for mask gating
    """
    if use_gating and gating_dict and current_gating_params:
        children = [html.B("Gating parameters (norm 0-1)\n",
                              style={"color": "black"}), html.Br(), html.Br()]
        for current_param in current_gating_params:
            try:
                children.append(html.Span(f"{str(current_param)}: "
                                      f"{gating_dict[current_param]['lower_bound']},   "
                                      f"{gating_dict[current_param]['upper_bound']}\n"))
                children.append(html.Br())
            except KeyError:
                pass
        if object_id_list is not None and isinstance(object_id_list, list):
            children.append(html.Span(f"{len(object_id_list)} objects\n"))
        return gating_label_zoom(children, zoom_obj_list)
    if from_custom_list and object_id_list:
        children = [html.B("Gating (custom ID list)\n",
                           style={"color": "black"}), html.Br(),
                    html.Span(f"{len(object_id_list)} objects")]
        return gating_label_zoom(children, zoom_obj_list)
    return []

def gating_label_zoom(children: list, zoom_obj_list: list=None):
    """
    Append an `html.Span` displaying the number of gated objects in the current zoom
    if ths lists exists.
    """
    if zoom_obj_list is not None:
        return children + [html.Br(),
                html.Span(f"\n({len(zoom_obj_list)} object(s) in view)")]
    return children

def limit_length_to_quantify(marker_list: list, size_threshold: int=1000):
    """
    Check the list of markers in the current session. If the list is longer than a set threshold,
    do not pass the list for quantification selection. Ensures that datasets with too many markers
    are not quantified in-app
    """
    return marker_list if len(marker_list) < size_threshold else []
