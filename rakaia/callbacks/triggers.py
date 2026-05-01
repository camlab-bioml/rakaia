from typing import Union
import numpy as np

def new_roi_same_dims(triggered_id: str, cur_dimensions: Union[tuple, list], first_image: np.array):
    """
    Do not update the canvas graph component with a new property if the dimensions of the incoming ROI
    are the same as the current graph
    """
    return cur_dimensions is not None and (first_image.shape[0] == cur_dimensions[0]) and \
    (first_image.shape[1] == cur_dimensions[1]) and triggered_id not in ["data-selection-refresh"]

def channel_already_added(triggered_id: str, triggered_list: Union[list, tuple], session_vars: dict):
    """
    Do not update the trigger if the channel options and the current selection haven't changed
    If an error occurs, play if safe and return False to re-trigger the callback
    """
    try:
        return triggered_id == "images_in_blend" and triggered_list[0]['value'] == session_vars["cur_channel"]
    except KeyError:
        return False

def no_canvas_mask(triggered_id: str, mask_selection: Union[str, None]=None,
                   apply_mask: Union[bool, list]=False):
    """
    Assess if the callback should fired if a mask-related input is triggered,
    but no mask has been imported
    """
    # Case 1: there is no mask available on mask param changes
    # Case 2: a mask is being selected, but applying the mask is not enabled
    return (triggered_id in ["mask-options", "mask-blending-slider", "add-mask-boundary",
        "add-cell-id-mask-hover", "apply-mask"] and not mask_selection and not apply_mask) or \
        (triggered_id == "mask-options" and not apply_mask)

def global_filter_disabled(triggered_id: str, global_apply_filter: Union[list, bool]=False):
    """
    Assess if the callback should fired if a global filter-related input is triggered,
    but the toggle for the global filter is off
    """
    return triggered_id in ["global-filter-type", "global-kernel-val-filter",
                                                  "global-sigma-val-filter"] and not global_apply_filter

def channel_order_as_default(triggered_id: str, channel_order: list, currently_selected: list):
    """
    Do not register the callback if the channel order triggers the callback,
    but the order has not changed
    """
    return triggered_id in ["channel-order"] and channel_order == currently_selected

def set_annotation_indices_to_remove(trigger_id: str, annotations: dict, data_selection: str,
                                     annot_table_indices: Union[list, None]=None):
    """
    Set the annotation indices to remove based on the trigger. Indices can come from a selection in the
    ROI table or from bulk clearing
    """
    if trigger_id == "clear-annotation_dict" and data_selection in annotations:
        return [int(i) for i in range(len(annotations[data_selection].keys()))]
    return annot_table_indices if trigger_id == "delete-annotation-tabular" else None

def reset_on_visium_spot_size_change(triggered_id: str, raw_data_dict: dict,
                                     layer_dict: dict, data_selection: str):
    """
    Reset the raw channel and currently channel blend layers when the visium spot size is changed.
    Changing the size of the visium spot requires all the raw and RGB layers for currently selected
    channels to be reconstructed.
    """
    if triggered_id == "spatial-spot-rad" and data_selection in raw_data_dict and \
            data_selection in layer_dict:
        raw_data_dict[data_selection] = {marker: None for marker in raw_data_dict[data_selection].keys()}
        layer_dict[data_selection] = {}
    return raw_data_dict, layer_dict

def no_channel_for_view(trigger: str, channel_selected: str, view_by_channel: Union[bool, list]):
    """
    Returns True if the conditions are met where the gallery view by channel mode is selected,
    but cannot be toggled, either because no channel is selected, or the feature hasn't been enabled
    """
    return ((trigger == "unique-channel-list" and not view_by_channel) or
            (trigger == "toggle-gallery-view" and not channel_selected))

def empty_slider_values(vals: Union[list, tuple, None]=None):
    """
    Check if the slider data object has empty values
    """
    return vals is not None and any(elem is None for elem in vals)

def use_channel_autofill(rgb_layers: dict, roi_selection: str,
                         channel: str, autofill_toggle: bool=True,
                         autofill_enabled: bool=True):
    """
    Check if the app should auto-assign a blend colour to a channel. Only can be used if:
    - The user has toggled on the autofill slider
    - The channel has not yet been added to the RGB layer cache
    - The internal app state allows for autofill. If the most previous action was changing the
    ROI, then it is disabled. Otherwise, if a user has selected a channel, it is enabled
    """
    return autofill_toggle and autofill_enabled and (
            channel not in rgb_layers[roi_selection].keys() or
            rgb_layers[roi_selection][channel] is None)


def layout_has_modified_shape(canvas_layout: Union[dict, None]=None):
    """
    Detect if the canvas layout has a modified shape (i.e. re-dragged)
    """
    return canvas_layout is not None and any('.path' in key for key in canvas_layout.keys())

def stitch_cache_delete(triggered_id: str, stitch_selection: Union[str, None]=None):
    """
    Determine if the user wants to delete the current stitch from the cache
    """
    return triggered_id == "stitch-image-delete" and stitch_selection is not None

def wsi_selection(triggered_id: str, target_id: str, image_cache: Union[dict, None]=None,
                  image_selection: Union[str, None]=None):
    """
    Detect which trigger was used to render a WSI, and check that the image selection is in the cache
    """
    return str(triggered_id) == str(target_id) and None not in \
        (image_cache, image_selection) and str(image_selection) in image_cache

def reset_image_blend_cache(triggered_id: str):
    """
    Detect which trigger is used to recreate the canvas image, and see if the trigger needs to recompute the image blend
    """
    return str(triggered_id) in ['canvas-layers', 'bool-apply-global-filter', 'global-filter-type', "global-kernel-val-filter", "global-sigma-val-filter"]

def reset_mask_fill_cache(triggered_id: str):
    """
    Detect which trigger is used to recreate the mask fill overlay, and see if the trigger needs to recompute the mask fill
    """
    return str(triggered_id) in ['mask-options', 'mask-blending-slider', 'toggle-cluster-annotations', 'cluster-colour-assignments-dict',
                            'cluster-col', 'cluster-annotation-type', 'apply-gating', 'gating-cell-list', 'cluster-label-selection']
