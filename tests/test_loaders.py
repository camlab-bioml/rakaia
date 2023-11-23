import dash_core_components as dcc
import numpy as np
import plotly.express as px
from ccramic.inputs.loaders import (
    wrap_child_in_loading,
    previous_roi_trigger,
    next_roi_trigger,
    reset_graph_data)

def test_loader_children():
    child = dcc.Store(id="test_store")
    component = wrap_child_in_loading(child, wrap=True)
    assert isinstance(component, dcc.Loading)
    component_2 = wrap_child_in_loading(child, wrap=False)
    assert isinstance(component_2, dcc.Store)

def test_blanking_graph_data():
    fake_image = np.full((100, 100, 3), 5)
    graph = px.imshow(fake_image)
    assert 'data' in graph
    assert graph['data'] is not None and len(graph['data']) > 0
    blanked = reset_graph_data(graph)
    assert len(blanked['data']) == 0
    assert blanked['layout'] == graph['layout']

def test_loaders_roi_triggers():
    assert previous_roi_trigger("prev-roi", 1, None, None)
    assert not previous_roi_trigger("prev-roi", 0, None, None)
    assert previous_roi_trigger("keyboard-listener", 0, {'keyCode': 37}, 1)
    assert not previous_roi_trigger("keyboard-listener", 0, {'keyCode': 37}, 0)
    assert not previous_roi_trigger("keyboard-listener", 1, {'badKey': 37}, 1)

    assert next_roi_trigger("next-roi", 1, None, None)
    assert not next_roi_trigger("next-roi", 0, None, None)
    assert next_roi_trigger("keyboard-listener", 0, {'keyCode': 39}, 1)
    assert not next_roi_trigger("keyboard-listener", 0, {'keyCode': 39}, 0)
    assert not next_roi_trigger("keyboard-listener", 1, {'badKey': 39}, 1)
