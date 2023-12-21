from ccramic.utils.shapes import *

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

    assert is_bad_shape(shape)

    shape = {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
             'x0': 1, 'x1': 3,
             'xref': 'x', 'y0': 2, 'y1': 4,
             'yref': 'y',
             'fillcolor': 'blue',
             'label': 'this is a label'}

    assert not is_bad_shape(shape)
