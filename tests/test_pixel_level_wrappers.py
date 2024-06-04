from ccramic.callbacks.pixel_level_wrappers import (
    parse_global_filter_values_from_json,
    parse_local_path_imports,
    mask_options_from_json)
import dash
import os

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
