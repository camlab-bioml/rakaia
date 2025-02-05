"""Module containing utility functions for managing shape lists in the main canvas
"""

from typing import Union
import plotly.graph_objs as go

def is_cluster_annotation_circle(shape):
    """
    Return if an annotation shape is a cluster circle. These are strictly defined as circles that are not editable,
    or a shape that does not have an editable property
    """
    return 'type' in shape and shape['type'] in ['circle'] and not shape['editable']

def is_bad_shape(shape):
    """
    Determine if a shape contained in the current canvas layout is malformed. Examples of malformed shapes:
    - Missing a`type` entry in the dictionary representation
    - Dictionary representation is None or empty (no length)
    """
    # has_texttemplate = shape is not None and 'label' in shape and 'texttemplate' in shape['label']
    # only_label = shape is not None and 'label' in shape and len(shape) == 1
    missing_type = 'type' not in shape or shape in [None, "None"] or len(shape) == 0
    return missing_type

def filter_annotation_shapes(canvas: Union[dict, go.Figure]):
    """
    Return a list of non-annotation shapes from a current canvas figure: Annotation shapes are editable shapes
    drawn on the canvas by the user
    Currently, keep only line shapes are those represent the scalebar when filtering out annotation shapes
    """
    new_shapes = []
    if 'layout' in canvas and 'shapes' in canvas['layout']:
        for shape in canvas['layout']['shapes']:
            if shape is not None and ('type' in shape and shape['type'] not in
                                      ['rect', 'path', 'circle'] and not is_bad_shape(shape)):
                new_shapes.append(shape)
    return new_shapes
