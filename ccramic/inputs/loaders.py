from dash import dcc

def wrap_child_in_loading(child, wrap=True, fullscreen=True, wrap_type="default"):
    """
    Wrap a child input in a dash loading screen if wrap is True. Otherwise, return the child as is
    """
    return dcc.Loading(child, fullscreen=fullscreen, type=wrap_type) if wrap else child

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
