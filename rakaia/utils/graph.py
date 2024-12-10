"""Module containing utility functions for maintaining proper component attributes
for the main canvas
"""

from typing import Union
import plotly.graph_objs as go
def strip_invalid_shapes_from_graph_layout(cur_graph: Union[go.Figure, dict]):
    """
    Remove any incorrectly formatted graph objects
    """
    # IMP: this check allows for channels to be added after shapes are drawn
    # removes shape properties that are added incorrectly
    if 'layout' in cur_graph and 'shapes' in cur_graph['layout'] and len(cur_graph['layout']['shapes']) > 0:
        cur_graph['layout']['shapes'] = [shape for shape in cur_graph['layout']['shapes'] if shape is not None and \
                                         'type' in shape]
    return cur_graph
