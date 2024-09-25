import numpy as np
from rakaia.io.gallery import (
    generate_roi_query_gallery_children,
    generate_channel_tile_gallery_children,
    set_channel_thumbnail,
    set_gallery_thumbnail_from_signal_retention,
    replace_channel_gallery_aliases)
from rakaia.utils.pixel import resize_for_canvas
import dash_bootstrap_components as dbc
from numpy.testing import assert_array_equal

def test_channel_thumbnail_signal_retention():
    # if the signal kept is low, keep the original array
    original_image = np.zeros((2500, 2500))
    original_image.fill(10000)
    down_sampled = resize_for_canvas(original_image)
    array_use = set_gallery_thumbnail_from_signal_retention(original_image, down_sampled, original_image,
                                                            (np.mean(down_sampled) / np.mean(original_image)))

    assert np.sum(array_use) - np.sum(original_image) <= 1.6e-10

    # signal retained is high enough
    original_image = np.zeros((2500, 2500))
    original_image.fill(100)
    down_sampled = resize_for_canvas(original_image)
    array_use = set_gallery_thumbnail_from_signal_retention(original_image, down_sampled, original_image,
                                                            (np.mean(down_sampled) / np.mean(original_image)))
    assert np.sum(array_use) - np.sum(down_sampled) <= 1.6e-10

    # dimension is too large so use the down-sample
    original_image = np.zeros((5000, 5000))
    original_image.fill(10)
    down_sampled = resize_for_canvas(original_image)
    array_use = set_gallery_thumbnail_from_signal_retention(original_image, down_sampled, original_image,
                                                            (np.mean(down_sampled) / np.mean(original_image)))
    assert np.sum(array_use) - np.sum(down_sampled) <= 1.6e-10

def test_generate_channel_gallery_children():
    """
    test that the function for returning the single channel images for one ROI returns the children
    comprised of columns and cards
    """
    gallery_dict = {"im_1": np.zeros((1000, 1000)), "im_2": np.zeros((1000, 1000)),
                    "im_3": np.zeros((1000, 1000)),
                    "im_4": np.zeros((1000, 1000)), "im_5": np.zeros((1000, 1000))}

    thumbnail, alternative = set_channel_thumbnail({}, gallery_dict["im_1"])
    assert thumbnail.shape == (400, 400)
    assert alternative.shape == (1000, 1000)
    # use a zoom feature for the gallery
    canvas_layout = {'xaxis.range[0]': 100, 'xaxis.range[1]': 200, 'yaxis.range[1]': 100, 'yaxis.range[0]': 200}
    zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[1]', 'yaxis.range[0]']
    thumbnail, alternative = set_channel_thumbnail(canvas_layout, gallery_dict["im_1"], zoom_keys, True)
    assert thumbnail.shape == alternative.shape == (100, 100)
    blend_colour_dict = {"im_1": {"color": "#0000FF", "x_lower_bound": None,
                                  "x_upper_bound": None, "filter_type": None, "filter_val": None},
                         "im_2": {"color": "#0000FF", "x_lower_bound": None,
                                  "x_upper_bound": None, "filter_type": None, "filter_val": None},
                         "im_3": {"color": "#0000FF", "x_lower_bound": None,
                                  "x_upper_bound": None, "filter_type": None, "filter_val": None},
                         "im_4": {"color": "#0000FF", "x_lower_bound": None,
                                  "x_upper_bound": None, "filter_type": None, "filter_val": None},
                         "im_5": {"color": "#0000FF", "x_lower_bound": None,
                                  "x_upper_bound": None, "filter_type": None, "filter_val": None}}
    aliases = {key: key for key in blend_colour_dict.keys()}
    row_children = generate_channel_tile_gallery_children(gallery_dict, canvas_layout, zoom_keys,
                                                          blend_colour_dict, None, None, aliases, 0,
                                                          toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    assert len(row_children) == len(gallery_dict)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)

    # assert that the gallery will still render if there is an index incompatibility

    blend_colour_dict = {"im_1": {"color": "#0000FF", "x_lower_bound": None,
                                  "filter_type": None, "filter_val": None},
                         "im_2": {"color": "#0000FF", "x_lower_bound": None,
                                  "filter_type": None, "filter_val": None},
                         "im_3": {"color": "#0000FF", "x_lower_bound": None,
                                  "filter_type": None, "filter_val": None},
                         "im_4": {"color": "#0000FF", "x_lower_bound": None,
                                  "filter_type": None, "filter_val": None},
                         "im_5": {"color": "#0000FF", "x_lower_bound": None,
                                  "filter_type": None, "filter_val": None}}

    canvas_layout = {'xaxis.range[0]': 100, 'xaxis.range[1]': 200, 'yaxis.range[0]': 200, 'yaxis.range[1]': 100}
    row_children = generate_channel_tile_gallery_children(gallery_dict, canvas_layout, zoom_keys,
                                                          blend_colour_dict, None, None, aliases, 0,
                                                          toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    assert len(row_children) == len(gallery_dict)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)

    # do not use a zoom feature for the channel gallery
    canvas_layout = {}
    row_children = generate_channel_tile_gallery_children(gallery_dict, canvas_layout, zoom_keys,
                                                          blend_colour_dict, None, None, aliases, 0,
                                                          toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    assert len(row_children) == len(gallery_dict)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)


def test_generate_roi_gallery_children():
    """
    test that the ROI query function returns a list of children comprised of dbc columns and cards
    """
    roi_dict = {"roi_1": np.zeros((100, 100, 3)), "roi_2": np.zeros((1000, 1000)), "roi_3": np.zeros((1000, 100, 3))}
    gallery_children, roi_list = generate_roi_query_gallery_children(roi_dict)
    assert len(gallery_children) == len(roi_dict)
    for elem in gallery_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)
    assert all([elem in roi_list for elem in roi_dict.keys()])

def test_recursive_gallery_children(recursive_gallery_children, recursive_aliases_2):
    names = []
    for child in recursive_gallery_children:
        names.append(child['props']['children'][0]['id'])
    for child in recursive_gallery_children:
        for sub_child in child['props']['children']:
            assert 'initial_label' in sub_child['children']
            assert 'rakaia' not in sub_child['children']
    recur_names = []
    edited_children = replace_channel_gallery_aliases(recursive_gallery_children, recursive_aliases_2)
    for child in edited_children:
        recur_names.append(child['props']['children'][0]['id'])
        for sub_child in child['props']['children']:
            assert 'initial_label' not in sub_child['children']
            assert 'rakaia' in sub_child['children']

    assert recur_names == names
    empty_children = replace_channel_gallery_aliases({}, recursive_aliases_2)
    assert empty_children == {}
    malformed = {"other": {"key_1": {"props": None}, "key_2": {"one": 1}}}
    assert replace_channel_gallery_aliases(malformed, recursive_aliases_2) == malformed == \
           {'other': {'key_1': {'props': None}, 'key_2': {'one': 1}}}
