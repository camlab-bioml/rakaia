import os
import tempfile
import pickle
import numpy as np
from PIL import Image
from rakaia.callbacks.pixel_wrappers import parse_steinbock_umap
from rakaia.io.gallery import (
    roi_query_gallery_children,
    channel_tile_gallery_children,
    set_channel_thumbnail,
    set_gallery_thumbnail_from_signal_retention,
    replace_channel_gallery_aliases,
    channel_tiles, gallery_export_template,
    channel_tiles_from_gallery, umap_pipeline_tiles,
    umap_gallery_children,
    rainbow_spectrum)
from rakaia.utils.pixel import resize_for_canvas, filter_by_upper_and_lower_bound
import dash_bootstrap_components as dbc

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
    tiles = channel_tiles(gallery_dict, canvas_layout, zoom_keys,
                        blend_colour_dict, None, None, aliases, 0,
                        toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    row_children = channel_tile_gallery_children(tiles)
    assert len(tiles) == len(gallery_dict) == len(row_children)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)
    # test exporting the channels to html

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "out_chan_gallery.html")
        assert not os.path.exists(file_path)
        export = gallery_export_template(file_path, tiles)
        assert os.path.exists(export)
        if os.access(export, os.W_OK):
            os.remove(export)
        assert not os.path.exists(file_path)


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
    tiles = channel_tiles(gallery_dict, canvas_layout, zoom_keys,
                                                 blend_colour_dict, None, None, aliases, 0,
                                                 toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    row_children = channel_tile_gallery_children(tiles)
    assert len(tiles) == len(gallery_dict) == len(row_children)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)

    # do not use a zoom feature for the channel gallery
    canvas_layout = {}
    tiles = channel_tiles(gallery_dict, canvas_layout, zoom_keys,
                                                 blend_colour_dict, None, None, aliases, 0,
                                                 toggle_gallery_zoom=True, toggle_scaling_gallery=True)
    row_children = channel_tile_gallery_children(tiles)
    assert len(tiles) == len(gallery_dict) == len(row_children)
    for elem in row_children:
        assert isinstance(elem, dbc.Col)
        assert isinstance(elem.children, dbc.Card)

def test_tiles_from_gallery_children(get_current_dir):
    with open(os.path.join(get_current_dir, 'rois.pickle'), 'rb') as f:
        channel_children = pickle.load(f)
        tiles_from_gal = channel_tiles_from_gallery(channel_children)
        assert len(tiles_from_gal) == 2
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "out_chan_gallery.html")
            assert not os.path.exists(file_path)
            export = gallery_export_template(file_path, tiles_from_gal)
            assert os.path.exists(export)
            if os.access(export, os.W_OK):
                os.remove(export)
            assert not os.path.exists(file_path)


def test_generate_roi_gallery_children():
    """
    test that the ROI query function returns a list of children comprised of dbc columns and cards
    """
    roi_dict = {"roi_1": np.zeros((100, 100, 3)), "roi_2": np.zeros((1000, 1000)), "roi_3": np.zeros((1000, 100, 3))}
    gallery_children, roi_list = roi_query_gallery_children(roi_dict)
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

def test_umap_gallery_tiles(get_current_dir):
    umap_files = parse_steinbock_umap(os.path.join(get_current_dir, 'steinbock', 'test_mcd'))
    umap_tiles = umap_pipeline_tiles(umap_files)
    assert len(umap_tiles) == len(umap_files) == 3
    for dist in [0, 0.1, 0.25]:
        assert any([str(dist) in file_base for file_base in list(umap_tiles.keys())])

    gallery_children_umap = umap_gallery_children(umap_tiles)
    assert len(gallery_children_umap) == len(umap_tiles)
    assert not umap_gallery_children({})

def test_rainbow_spectrum(get_current_dir):
    greyscale_image = np.array(Image.open(os.path.join(get_current_dir, "for_recolour.tiff")))
    filtered = filter_by_upper_and_lower_bound(greyscale_image, 0,
                                               np.percentile(greyscale_image, 99))
    rainbow = rainbow_spectrum(filtered)
    assert np.array_equal(rainbow[161, 90], np.array([255, 0, 0]))
    assert np.array_equal(rainbow[110, 59], np.array([0, 0, 0]))
    assert np.array_equal(rainbow[111, 46], np.array([121, 9, 254]))
