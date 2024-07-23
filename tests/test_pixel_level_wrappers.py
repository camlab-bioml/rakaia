from rakaia.callbacks.pixel_wrappers import (
    parse_global_filter_values_from_json,
    parse_local_path_imports,
    mask_options_from_json,
    bounds_text,
    generate_annotation_list)
import dash
import os

def test_generate_annotation_lists():
    layout_1 = {'xaxis.range[0]': 259.7134146341464, 'xaxis.range[1]': 414.2865853658536,
                'yaxis.range[0]': 363.3457507621951, 'yaxis.range[1]': 208.7725800304878}
    annot_list_1 = generate_annotation_list(layout_1)
    assert len(annot_list_1) == 1
    assert list(annot_list_1.values())[0] == "zoom"
    layout_2 = {'shapes': [{'editable': True, 'fillcolor': 'rgba(0, 0, 0, 0)', 'fillrule': 'evenodd', 'layer': 'above',
                'line': {'color': 'white', 'dash': 'solid', 'width': 4}, 'opacity': 1,
                'path': 'M251.48170731707316,187.73599466463418L204.83536585365857,180.41892149390247L'
                        '181.05487804878052,181.33355564024393L161.84756097560978,192.30916539634148L'
                        '152.70121951219514,204.19940929878052L146.2987804878049,220.66282393292684L'
                        '149.0426829268293,239.87014100609755L152.70121951219514,253.5896532012195L'
                        '164.59146341463418,268.2237995426829L177.39634146341467,274.6262385670732L'
                        '213.06707317073173,276.4555068597561L234.10365853658539,269.1384336890244L'
                        '248.7378048780488,252.67501905487805L260.6280487804878,240.78477515243904L'
                        '299.9573170731707,233.46770198170734L311.8475609756098,227.97989710365857L'
                        '321.9085365853659,217.004287347561L321.9085365853659,197.79697027439028L'
                        '317.3353658536585,189.5652629573171Z', 'type': 'path', 'xref': 'x', 'yref': 'y'},
                {'editable': True, 'fillcolor': 'rgba(0, 0, 0, 0)', 'fillrule': 'evenodd', 'layer': 'above', 'line':
                {'color': 'white', 'dash': 'solid', 'width': 4}, 'opacity': 1, 'type': 'rect',
                 'x0': 417.9451219512195, 'x1': 535.9329268292684, 'xref': 'x', 'y0': 141.08965320121953,
                 'y1': 233.46770198170734, 'yref': 'y'}, {'line': {'color': 'white', 'width': 2}, 'type': 'line',
                'x0': 0.8350000000000001, 'x1': 0.935, 'xref': 'paper', 'y0': 0.05, 'y1': 0.05, 'yref': 'paper'},
                {'editable': True, 'label': {'text': '', 'texttemplate': ''}, 'xref': 'x', 'yref': 'y',
                 'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                 'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type': 'rect',
                 'x0': 210.32317073170734, 'y0': 334.9920922256098, 'x1': 253.3109756097561, 'y1': 454.8091653963415}]}
    annot_list_2 = generate_annotation_list(layout_2, True)
    assert len(annot_list_2) == 3
    assert list(annot_list_2.values()) == ["path", "rect", "rect"]
    only_last = generate_annotation_list(layout_2, False)
    assert len(only_last) == 1
    assert list(only_last.keys())[0] == list(annot_list_2.keys())[-1]

def test_parse_global_filters():
    config_dict = {"config": {"filter": {"global_apply_filter": False,
                                        "global_filter_type": "median", "global_filter_val": 5,
                                        "global_filter_sigma": 1}}}

    global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma = \
        parse_global_filter_values_from_json(config_dict['config'])

    assert not global_apply_filter
    assert global_filter_type == "median"
    assert global_filter_val == 5
    assert global_filter_sigma == 1

    config_dict = {"config": {"fake_key": {"global_apply_filter": False,
                                         "global_filter_type": "median", "global_filter_val": 5,
                                         "global_filter_sigma": 1}}}

    globals = parse_global_filter_values_from_json(config_dict['config'])

    assert all([isinstance(elem, dash._callback.NoUpdate) for elem in globals])

    config_dict = {"config": {"filter": {"global_apply_filter": False,
                                         "missing_key": "median", "global_filter_val": 5,
                                         "global_filter_sigma": 1}}}

    globals = parse_global_filter_values_from_json(config_dict['config'])
    assert all([isinstance(elem, dash._callback.NoUpdate) for elem in globals])

def test_import_paths_from_local(get_current_dir):
    mcd_file = os.path.join(get_current_dir, 'query.mcd')
    imports, error = parse_local_path_imports(mcd_file, {'uploads': []}, {"error": None})
    assert imports['uploads']
    assert 'query.mcd' in imports['uploads'][0]
    assert isinstance(error, dash._callback.NoUpdate)
    imports, error = parse_local_path_imports(get_current_dir, {'uploads': []}, {"error": None})
    assert imports['uploads']
    assert isinstance(error, dash._callback.NoUpdate)
    imports, error = parse_local_path_imports('', {'uploads': []}, {"error": None})
    assert error['error']
    assert isinstance(imports, dash._callback.NoUpdate)

def test_parse_json_mask_options():
    config_dict = {"mask": {"mask_toggle": True, "mask_level": 62.5, "mask_boundary": [" add boundary"], "mask_hover": []}}
    assert mask_options_from_json(config_dict) == list(config_dict['mask'].values())
    assert all([isinstance(elem, dash._callback.NoUpdate) for elem in mask_options_from_json({})])

def test_bounds_text():
    children, bounds = bounds_text(0.0, 10, 0.0, 10)
    assert len(children) == 3
    assert children[1].children == 'Current bounds: \n X: (0.0, 10), Y: (0.0, 10)'
    assert bounds == {'x_low': 0.0, 'x_high': 10, 'y_low': 0.0, 'y_high': 10}
    children, bounds = bounds_text(None, 100, None, 100)
    assert not children and not bounds
