

def strip_invalid_shapes_from_graph_layout(cur_graph):
    """
    Remove any incorrectly formatted graph objects
    """
    try:
        # IMP: this check allows for channels to be added after shapes are drawn
        # removes shape properties that are added incorrectly
        if 'shapes' in cur_graph['layout'] and len(cur_graph['layout']['shapes']) > 0:
            for shape in cur_graph['layout']['shapes']:
                if 'label' in shape and 'texttemplate' in shape['label']:
                    del shape['label']['texttemplate']
            cur_graph['layout']['shapes'] = [shape for shape in cur_graph['layout']['shapes'] if shape is not None]
    except (KeyError, TypeError):
        pass
    return cur_graph
