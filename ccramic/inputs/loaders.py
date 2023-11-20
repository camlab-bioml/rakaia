
import dash_core_components as dcc
def wrap_child_in_loading(child, wrap=True, fullscreen=True, wrap_type="default"):
    """
    Wrap a child input in a dash loading screen if wrap is True. Otherwise, return the child as is
    """
    return dcc.Loading(child, fullscreen=fullscreen, type=wrap_type) if wrap else child
