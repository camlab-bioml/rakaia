from ccramic.callbacks.pixel_level_wrappers import parse_global_filter_values_from_json
import dash

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
