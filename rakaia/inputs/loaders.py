from dash import dcc
from rakaia.io.session import SessionTheme
from typing import Union
from rakaia.utils.pixel import split_string_at_pattern


def wrap_child_in_loading(child, wrap=True, fullscreen=True, wrap_type="default"):
    """
    Wrap a child input in a dash loading screen if wrap is True. Otherwise, return the child as is
    """
    return dcc.Loading(child, fullscreen=fullscreen, type=wrap_type,
                       color=SessionTheme().widget_colour) if wrap else child

def reset_graph_data(graph):
    """
    Reset graph data while retaining the current layout parameters. Useful for "blanking" a figure
    """
    if 'data' in graph:
        graph['data'] = []
    return graph


def previous_roi_trigger(triggered_id, button_click, key_listener, key_events):
    """
    Detect if the app should navigate to the previous ROI based on either a button click or keyboard event
    """
    try:
        return (triggered_id == "prev-roi" and button_click > 0) or \
                (triggered_id == "keyboard-listener" and key_listener['keyCode'] == 37 and key_events > 0)
    except KeyError:
        return False

def next_roi_trigger(triggered_id, button_click, key_listener, key_events):
    """
    Detect if the app should navigate to the previous ROI based on either a button click or keyboard event
    """
    try:
        return (triggered_id == "next-roi" and button_click > 0) or \
                (triggered_id == "keyboard-listener" and key_listener['keyCode'] == 39 and key_events > 0)
    except KeyError:
        return False

def adjust_option_height_from_list_length(list_options, dropdown_type="image"):
    """
    Return an option height for the dropdown menu based on the maximum length of the list
    Ensures that long strings in `dcc.Dropdown` menus are sufficiently spaced vertically for readability
    """
    char_limits = {"image": 50, "mask": 40}
    if any([len(elem) >= char_limits[dropdown_type] for elem in list_options]):
        height_update = 100
    else:
        height_update = 50
    return height_update

def set_roi_tooltip_based_on_length(data_selection: str, delimiter: str="+++",
                                    char_threshold: Union[int, float]=45):
    if data_selection and len(data_selection) >= char_threshold:
        exp, slide, roi_name = split_string_at_pattern(data_selection, pattern=delimiter)
        roi_name = str(roi_name) + f" ({str(exp)})" if "acq" in str(roi_name) else str(roi_name)
        return f"Current ROI: {roi_name}"
    return None
