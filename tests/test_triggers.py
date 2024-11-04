from rakaia.callbacks.triggers import (
    new_roi_same_dims,
    no_canvas_mask,
    global_filter_disabled,
    channel_order_as_default,
    channel_already_added,
    set_annotation_indices_to_remove,
    reset_on_visium_spot_size_change,
    no_channel_for_view)
import numpy as np

def test_triggers():
    # if the assertion is correct, do not allow the callback to occur

    image = np.zeros((1000, 1000))
    assert new_roi_same_dims("new-image", (1000, 1000), image)
    assert not new_roi_same_dims("new-image", (999, 1000), image)
    assert not new_roi_same_dims("data-selection-refresh", (1000, 1000), image)

    assert channel_already_added("images_in_blend", [{"value": "first_channel"}], {"cur_channel": "first_channel"})
    assert not channel_already_added("images_in_blend", [{"value": "first_channel"}], {"cur_channel": "second_channel"})
    assert not channel_already_added("images_in_blend", [{"value": "first_channel"}], {"bad_key": "first_channel"})

    assert no_canvas_mask("mask-options", None)
    assert not no_canvas_mask("apply_mask", "mask_1")

    assert no_canvas_mask("mask-options", "mask_1", False)
    assert not no_canvas_mask("mask-options", "mask_1", True)

    assert global_filter_disabled("global-kernel-val-filter", [])
    assert not global_filter_disabled("global-filter-type", True)

    assert channel_order_as_default("channel-order", ["1", "2", "3"], ["1", "2", "3"])
    assert not channel_order_as_default("channel-order", ["1", "3", "2"], ["1", "2", "3"])
    assert not channel_order_as_default("diff", ["1", "2", "3"], ["1", "2", "3"])

    assert no_channel_for_view("unique-channel-list", "chan-1", False)
    assert not no_channel_for_view("unique-channel-list", "chan-1", True)
    assert no_channel_for_view("toggle-gallery-view", None, True)
    assert not no_channel_for_view("toggle-gallery-view", "chan-1", True)


def test_annotation_index_triggers():
    annot_dict = {"roi_1": {"annot_1": {"imported": True}, "annot_2": {"imported": True}, "annot_3": {"imported": True}}}
    indices_remove = set_annotation_indices_to_remove("clear-annotation_dict", annot_dict,
                                                      "roi_1", [])
    assert indices_remove == [0, 1, 2]
    indices_remove = set_annotation_indices_to_remove("delete-annotation-tabular", annot_dict,
                                                      "roi_1", [2])
    assert indices_remove == [2]
    indices_remove = set_annotation_indices_to_remove("delete-annotation-tabular", annot_dict,
                                                      "roi_1", [])
    assert not indices_remove

def test_reset_layers_on_visium_spot_change():
    raw = {"visium_1": {"marker_1": np.zeros((100, 100)),
                        "marker_2": np.zeros((100, 100))}}
    layers = {"visium_1": {"marker_2": np.ones((100, 100))}}
    raw_back, layers_back = reset_on_visium_spot_size_change("spatial-spot-rad", raw, layers, "visium_1")
    assert all(elem is None for elem in raw_back['visium_1'].values())
    assert not layers_back['visium_1']
    raw_same, layers_same = reset_on_visium_spot_size_change("diff_trigger", raw, layers, "visium_1")
    assert raw_same == raw
    assert layers_same == layers
