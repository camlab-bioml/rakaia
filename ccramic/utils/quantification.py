import numpy as np
from ccramic.utils.cell_level_utils import validate_mask_shape_matches_image
from ccramic.utils.pixel_level_utils import split_string_at_pattern
import pandas as pd
from dash import html

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
            expr_mat[:, (cell - offset)] = np.mean(image[is_cell])
        # cell_ids_adata = [f"{image_name}_{str(cell_id)}" for cell_id in cell_ids]

        # adata = ad.AnnData(
        #     X=expr_mat,
        #     obs=pd.DataFrame(index=cell_ids_adata, data={"image_name": image_name}),
        #     var=pd.DataFrame(index=channel_names)
        # )

        return expr_mat
    else:
        return None

def quantify_roi_xy_coordinates_area(mask):
    """
    Quantify the xy coordinates and area for every object in an ROI mask
    Returns three arrays of values that can be cast to dataframe series: x, y, and area
    """
    # reshape the image to retain only the first two dimensions
    mask = mask.reshape((mask.shape[0], mask.shape[1]))
    vals_dict = {"x": None, "y": None, "area": None}
    # Get unique list of cell IDs. Remove '0' which corresponds to background
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids != 0]
    # IMP: the cell ids start at 1, but the array positions start at 0, so offset the array positions by 1
    offset = 1 if min(cell_ids) == 1 else 0
    for key in vals_dict.keys():
        vals_dict[key] = np.zeros((1, len(cell_ids)))
    for cell in cell_ids:
        subset = np.where(mask == cell)
        vals_dict['x'][:, (cell - offset)] = subset[1].mean()
        vals_dict['y'][:, (cell - offset)] = subset[0].mean()
        vals_dict['area'][:, (cell - offset)] = subset[0].shape[0]
    return vals_dict['x'], vals_dict['y'], vals_dict['area']


def quantify_multiple_channels_per_roi(channel_dict, mask, data_selection, channels_to_quantify, aliases=None,
                                       dataset_options=None, delimiter: str="+++", mask_name: str=None):
    """
    Quantify multiple channels for a single ROI and concatenate into a dataframe with cells
    as rows and channels + metadata as columns
    """
    exp, slide, roi_name = split_string_at_pattern(data_selection, pattern=delimiter)
    channel_frame = pd.DataFrame()
    for channel in channels_to_quantify:
        if channel in list(channel_dict[data_selection].keys()):
            mat = quantify_one_channel(channel_dict[data_selection][channel], mask)
            column = pd.Series(mat.flatten())
            # set the channel label if the aliases exists with a different name
            channel_name = aliases[channel] if aliases is not None and channel in aliases.keys() else channel
            channel_frame[channel_name] = column
    roi_name_sample = roi_name
    if dataset_options is not None:
        for dataset in dataset_options:
            exp, slide, roi = split_string_at_pattern(dataset, pattern=delimiter)
            if roi == roi_name:
                index = dataset_options.index(dataset) + 1
                roi_name_sample = f"{exp}_{index}"
    # TODO: change the order of the identifying columns here, and set the description to the mask used for the quant
    # to ensure matching
    channel_frame['description'] = mask_name
    channel_frame['cell_id'] = pd.Series(range(1, (len(channel_frame.index) + 1)), dtype='int64')
    channel_frame['sample'] = roi_name_sample
    x, y, area = quantify_roi_xy_coordinates_area(mask)
    channel_frame['x'] = pd.Series(x.flatten())
    channel_frame['y'] = pd.Series(y.flatten())
    channel_frame['area'] = pd.Series(area.flatten())
    return channel_frame


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
        else:
            return new_frame
    else:
        if empty_master_frame and not empty_new_frame:
            return new_frame
        else:
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
                                          gating_vals: list=[0.0, 1.0]):
    """
    Update the current gating dictionary with the range slider values for a specific parameter from
    the gating modifier dropdown
    """
    current_gate_dict = {gate_selected: {}} if current_gate_dict is None else current_gate_dict
    if gate_selected not in current_gate_dict: current_gate_dict[gate_selected] = \
        {'lower_bound': None, 'upper_bound': None}
    # if current_gate_dict[gate_selected]['lower_bound'] != float(min(gating_vals)) and \
    #         current_gate_dict[gate_selected]['upper_bound'] != float(max(gating_vals)):
    current_gate_dict[gate_selected]['lower_bound'] = float(min(gating_vals))
    current_gate_dict[gate_selected]['upper_bound'] = float(max(gating_vals))
    return current_gate_dict

def gating_label_children(use_gating: bool = True, gating_dict: dict = None,
                          current_gating_params: list=None):
    """
    Generate the HTML legend for the current parameters used for mask gating
    """
    if use_gating and gating_dict and current_gating_params:
        children = [html.B("Current gating parameters (norm 0-1)\n",
                              style={"color": "black"}), html.Br(), html.Br()]
        for current_param in current_gating_params:
            try:
                children.append(html.Span(f"{str(current_param)}: "
                                      f"{gating_dict[current_param]['lower_bound']},   "
                                      f"{gating_dict[current_param]['upper_bound']}\n"))
                children.append(html.Br())
            except KeyError:
                pass
        return children
    return []
