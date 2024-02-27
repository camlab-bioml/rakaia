
from ccramic.utils.filter import (
    return_current_or_default_filter_apply,
    return_current_or_default_filter_param,
    return_current_channel_blend_params,
    return_current_or_default_channel_color,
    return_current_default_params_with_preset)
import dash

def test_basic_filter_parsers():
    blend_dict = {"channel_1": {"color": '#FFFFFF', "filter_type": "median", "filter_val": 3,  "filter_sigma": 1.0}}
    fil_type, fil_val, fil_sigma, chan_color = return_current_channel_blend_params(blend_dict, "channel_1")
    assert fil_type == "median"
    assert fil_val == 3
    assert fil_sigma == 1.0
    assert chan_color == '#FFFFFF'
    assert return_current_channel_blend_params(blend_dict, "channel_2") == (None, None, None, None)

    assert isinstance(return_current_or_default_filter_apply(True, "median", 3, 1.0), dash._callback.NoUpdate)
    assert return_current_or_default_filter_apply(False, "median", 3, 1.0) == [' Apply/refresh filter']

    assert return_current_or_default_filter_param(3, 1) == 1
    assert isinstance(return_current_or_default_filter_param(3, 3), dash._callback.NoUpdate)

    assert return_current_or_default_filter_param("median", "gaussian") == "gaussian"
    assert isinstance(return_current_or_default_filter_param("gaussian", "gaussian"), dash._callback.NoUpdate)

    assert return_current_or_default_channel_color({'hex': '#FFFFFF'}, '#FF0000') == {'hex': '#FF0000'}
    assert isinstance(return_current_or_default_channel_color({'hex': '#FF0000'}, '#FF0000'), dash._callback.NoUpdate)


def test_filter_parsers_with_preset():
    to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return = \
        return_current_default_params_with_preset(None, None, None, None)
    assert not to_apply_filter
    assert filter_type_return == "median"
    assert filter_val_return == 3
    assert filter_sigma_return == 1.0
    assert isinstance(color_return, dash._callback.NoUpdate)

    to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return = \
        return_current_default_params_with_preset("gaussian", 5, 0.8, '#FF0000')
    assert to_apply_filter
    assert filter_type_return == "gaussian"
    assert filter_val_return == 5
    assert filter_sigma_return == 0.8
    assert color_return == {'hex': '#FF0000'}
