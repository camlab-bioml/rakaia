
def is_cluster_annotation_circle(shape):
    """
    Return if an annotation shape is a cluster circle. These are strictly defined as circles that are not editable,
    or a shape that does not have an editable property
    """
    return 'type' in shape and shape['type'] in ['circle'] and not shape['editable']

def is_bad_shape(shape):
    # has_texttemplate = shape is not None and 'label' in shape and 'texttemplate' in shape['label']
    only_label = shape is not None and 'label' in shape and len(shape) == 1
    return only_label
