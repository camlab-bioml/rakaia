from ccramic.components.layout import register_app_layout
from dash import html

def test_register_component_layout():
    app_config = {'auto_open': True, 'port': 5000,
                           'use_local_dialog': False,
                           'use_loading': False, 'persistence': True}
    app_layout = register_app_layout(app_config, "/tmp/")
    assert isinstance(app_layout, html.Div)
    assert len(app_layout.children) > 0
    app_children = []
    for child in app_layout.children:
        try:
            app_children.append(str(child.id))
        except AttributeError:
            pass
    assert 'uploaded_dict' in app_children
    assert 'keyboard-listener' in app_children
