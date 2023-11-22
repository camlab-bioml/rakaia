import dash_core_components as dcc
import plotly.graph_objs as go
from plotly.graph_objs.layout import YAxis, XAxis

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
