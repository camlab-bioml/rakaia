import numpy as np
from rakaia.io.gallery import generate_roi_query_gallery_children, generate_channel_tile_gallery_children
import dash_bootstrap_components as dbc


def test_generate_channel_gallery_children():
    """
    test that the function for returning the single channel images for one ROI returns the children
    comprised of columns and cards
    """
    gallery_dict = {"im_1": np.zeros((1000, 1000)), "im_2": np.zeros((1000, 1000)),
                    "im_3": np.zeros((1000, 1000)),
                  "im_4": np.zeros((1000, 1000)), "im_5": np.zeros((1000, 1000))}

    # use a zoom feature for the gallery
    canvas_layout = {'xaxis.range[0]': 100, 'xaxis.range[1]': 200, 'yaxis.range[1]': 100, 'yaxis.range[0]': 200}
    zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[1]', 'yaxis.range[0]']
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
