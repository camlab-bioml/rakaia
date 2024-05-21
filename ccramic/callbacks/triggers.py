from typing import Union
import numpy as np
"""
Triggers are responsible for detecting the context input trigger for callbacks and assessing whether
callbacks should fire
"""
def new_roi_same_dims(triggered_id: str, cur_dimensions: Union[tuple, list], first_image: np.array):
    """
    Do not update the canvas graph component with a new property if the dimensions of the incoming ROI
    are the same as the current graph
    """
    return cur_dimensions is not None and (first_image.shape[0] == cur_dimensions[0]) and \
    (first_image.shape[1] == cur_dimensions[1]) and triggered_id not in ["data-selection-refresh"]

def channel_already_added(triggered_id: str, triggered_list: Union[list, tuple], session_vars: dict):
    """
    Do not update if the trigger if the channel options and the current selection hasn't changed
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
