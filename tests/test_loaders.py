import dash_core_components as dcc
from ccramic.inputs.loaders import wrap_child_in_loading

def test_loader_children():
    child = dcc.Store(id="test_store")
    component = wrap_child_in_loading(child, wrap=True)
    assert isinstance(component, dcc.Loading)
    component_2 = wrap_child_in_loading(child, wrap=False)
    assert isinstance(component_2, dcc.Store)
