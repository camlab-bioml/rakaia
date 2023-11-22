import dash_core_components as dcc
from ccramic.inputs.loaders import wrap_child_in_loading, previous_roi_trigger, next_roi_trigger

def test_loader_children():
    child = dcc.Store(id="test_store")
    component = wrap_child_in_loading(child, wrap=True)
    assert isinstance(component, dcc.Loading)
    component_2 = wrap_child_in_loading(child, wrap=False)
    assert isinstance(component_2, dcc.Store)

def test_loaders_roi_triggers():
    assert previous_roi_trigger("prev-roi", 1, None, None)
    assert not previous_roi_trigger("prev-roi", 0, None, None)
    assert previous_roi_trigger("keyboard-listener", 0, {'keyCode': 37}, 1)
    assert not previous_roi_trigger("keyboard-listener", 0, {'keyCode': 37}, 0)

    assert next_roi_trigger("next-roi", 1, None, None)
    assert not next_roi_trigger("next-roi", 0, None, None)
    assert next_roi_trigger("keyboard-listener", 0, {'keyCode': 39}, 1)
    assert not next_roi_trigger("keyboard-listener", 0, {'keyCode': 39}, 0)
