from rakaia.callbacks.triggers import (
    new_roi_same_dims,
    no_canvas_mask,
    global_filter_disabled,
    channel_order_as_default,
    channel_already_added,
    set_annotation_indices_to_remove)
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
