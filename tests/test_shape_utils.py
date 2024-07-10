from ccramic.utils.shapes import (
    is_bad_shape,
    is_cluster_annotation_circle,
    filter_annotation_shapes)

def test_recognition_of_cluster_circle():
    shape = {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
                     'x0': 1, 'x1': 3,
                     'xref': 'x', 'y0': 2, 'y1': 4,
                     'yref': 'y',
                     'fillcolor': 'blue'}
    assert is_cluster_annotation_circle(shape)

    shape = {'editable': True, 'line': {'color': 'white'}, 'type': 'circle',
             'x0': 1, 'x1': 3,
             'xref': 'x', 'y0': 2, 'y1': 4,
             'yref': 'y',
             'fillcolor': 'blue'}
    assert not is_cluster_annotation_circle(shape)

def test_recognition_malformed_shape():
    shape = {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
                     'x0': 1, 'x1': 3,
                     'xref': 'x', 'y0': 2, 'y1': 4,
                     'yref': 'y',
                     'fillcolor': 'blue'}

    assert not is_bad_shape(shape)

    shape = {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
             'x0': 1, 'x1': 3,
             'xref': 'x', 'y0': 2, 'y1': 4,
             'yref': 'y',
             'fillcolor': 'blue',
             'label': {'texttemplate': ''}}

    assert not is_bad_shape(shape)

    shape = {'label': {'texttemplate': ''}}

    assert is_bad_shape(shape)

def test_filter_annotation_shapes():
    graph = {'layout': {'shapes': []}}
    assert not filter_annotation_shapes(graph)
    graph = {'layout': {'shapes': [{'type': 'circle'}, {'type': 'line'}]}}
    assert filter_annotation_shapes(graph) == [{'type': 'line'}]
    graph_2 = {'layout': {'shapes': [{'type': 'line'}, {'type': 'line'}]}}
    assert filter_annotation_shapes(graph_2) == graph_2['layout']['shapes']
    assert not filter_annotation_shapes({'layout': {'shapes': [{'type': 'circle'}, 'shape']}})
