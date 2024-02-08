
import plotly.graph_objs as go
from ccramic.utils.graph_utils import strip_invalid_shapes_from_graph_layout

def test_graph_strip_improper_shapes():
    fig = go.Figure()
    fig.add_shape(type="rect",
                  x0=3, y0=1, x1=6, y1=2,
                  line=dict(
                      color="RoyalBlue",
                      width=2,
                  ),
                  fillcolor="LightSkyBlue",
                  )
    assert len(fig['layout']['shapes']) > 0
    fig = strip_invalid_shapes_from_graph_layout(fig.to_dict())
    for shape in fig['layout']['shapes']:
        if 'label' in shape:
            assert 'texttemplate' not in shape['label']

    fake_graph = {}
    fake_graph = strip_invalid_shapes_from_graph_layout(fake_graph)
    assert len(fake_graph) == 0
