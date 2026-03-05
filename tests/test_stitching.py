import tempfile
import os
import numpy as np
import PIL
import pytest
from rakaia.stitch import (
    add_new_stitch,
    update_stitch_cache_with_blend,
    stitch_cache_dropdown_labels,
    download_stitch_image,
    stitch_image_preview)
from rakaia.stitch.mcd import (
    MCDAcqCoordinateParser,
    stitch_mcd_blends_from_gallery,
    set_gallery_mcd_rois_to_stitch,
    roi_identifier_to_steinbock_id,
    cur_roi_slide_matches_stitch)

def test_stitch_cache_update(stitch_cache):

    new_stitch = add_new_stitch(None, None, 100, 100)
    assert not new_stitch
    new_stitch = add_new_stitch(None, "stitch_1", 100, 100)
    assert len(new_stitch) == 1

    # if you try to add an image that goes beyond the limits, leave as is
    stitch_cache = update_stitch_cache_with_blend(stitch_cache, 'test_2',
                                                  2000, 2000,
                                                  np.ones((100, 100, 3)))
    assert all(np.sum(arr) == 0 for arr in stitch_cache.values())
    # update one stitch
    stitch_cache = update_stitch_cache_with_blend(stitch_cache, 'test_2',
                                                  5, 10,
                                                  np.ones((100, 100, 3)))
    assert np.sum(stitch_cache['test_1']) == 0
    assert np.sum(stitch_cache['test_2']) > 0
    # update another while retaining first edit
    stitch_cache = update_stitch_cache_with_blend(stitch_cache, 'test_1',
                                                  50, 60,
                                                  np.ones((20, 20, 3)))
    assert 0 < np.sum(stitch_cache['test_1']) < np.sum(stitch_cache['test_2'])

def test_stitch_cache_labels(stitch_cache):
    assert not stitch_cache_dropdown_labels(None)
    labels = stitch_cache_dropdown_labels(stitch_cache)
    assert len(labels) == 2
    for lab in labels:
        if lab['value'] == 'test_2':
            assert lab['label'] == 'test_2 (400x200)'

def test_stitch_download(stitch_cache):
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "stitch.zip")
        assert not os.path.exists(file_path)
        download_stitch = download_stitch_image(os.path.join(tmpdirname, "stitch"),
                           stitch_cache, 'test_2')
        assert str(download_stitch) == file_path
        assert os.path.exists(file_path)
        if os.access(download_stitch, os.W_OK):
            os.remove(download_stitch)

def test_stitch_image_preview(stitch_cache):
    visible, child = stitch_image_preview(stitch_cache, 'None')
    assert not visible
    assert not child
    visible, child = stitch_image_preview(stitch_cache, 'test_1')
    assert visible
    assert len(child) > 0
    assert isinstance(child[0].children.children[0].src, PIL.Image.Image)


def test_stitch_roi_from_mcd(get_current_dir):
    session_cache = {'uploads': [os.path.join(get_current_dir, 'query.mcd'),
                                 os.path.join(get_current_dir, 'for_quant.tiff')]}
    from_mcd, path = MCDAcqCoordinateParser(session_cache, 'query+++slide1+++Xylene_5').get_mcd_status()
    assert from_mcd
    assert str(path) == str(os.path.join(get_current_dir, 'query.mcd'))
    from_mcd, path = MCDAcqCoordinateParser(session_cache, None).get_mcd_status()
    assert not from_mcd
    from_mcd, path = MCDAcqCoordinateParser(session_cache, 'malformed_roi', '---').get_mcd_status()
    assert path is None

    # TODO: need to check that these tests actually parse out correct values rather than just semantic checks
    slide_width, slide_height = MCDAcqCoordinateParser(session_cache, 'query+++slide1+++Xylene_5').get_roi_slide_boundary_point()
    assert all(elem > 0 for elem in (slide_width, slide_height))

    x_start, y_start = MCDAcqCoordinateParser(session_cache,'query+++slide1+++Xylene_5').get_roi_coord_min()
    assert all(elem > 0 for elem in (x_start, y_start))

    # not MCD
    assert MCDAcqCoordinateParser(session_cache,
            'for_quant+++slideNA+++acq').get_roi_slide_boundary_point() == (None, None)
    assert MCDAcqCoordinateParser(session_cache,
            'for_quant+++slideNA+++acq').get_roi_coord_min() == (None, None)

    with (pytest.raises(TypeError)): MCDAcqCoordinateParser(session_cache, 'query+++slide1+++Xylene_5'
                                    ).get_roi_slide_boundary_point(bound_type=sum)


def test_roi_identifier_conversion(stitch_cache):
    """
    Conversion between the ROI identifier to steinbock ID is required for stitching from the gallery
    """
    assert roi_identifier_to_steinbock_id("query++++slide1++++first_roi_2") == "query_002"
    assert roi_identifier_to_steinbock_id("query++++slide1++++roi_10") == "query_010"
    assert roi_identifier_to_steinbock_id("query++++slide1++++roi_100") == "query_100"
    assert not roi_identifier_to_steinbock_id("query++++slide1++++roi")

def test_set_mcd_gallery_stitch(get_current_dir):
    indices, query_list, exclude = set_gallery_mcd_rois_to_stitch(
        ['query+++slide1+++PAP_1', 'query++++slide1+++Xylene_5'], None)
    assert len(indices['names']) == 2
    indices, query_list, exclude = set_gallery_mcd_rois_to_stitch(
        ['query+++slide1+++acq', 'query2++++slide1+++acq'], None)
    assert len(indices['names']) == 0 and not query_list

    mcd_filepath = [os.path.join(get_current_dir, 'steinbock', 'test_mcd', 'mcd', 'test.mcd')]
    slide_width, slide_height = MCDAcqCoordinateParser(mcd_filepath,
                                'test+++slide1+++chr10-h54h54-Gd158_2_18').get_roi_slide_boundary_point()
    new_stitch = {'test_stitch': np.zeros((slide_height, slide_width, 3))}
    assert np.sum(new_stitch['test_stitch']) == 0
    new_stitch = stitch_mcd_blends_from_gallery(new_stitch, 'test_stitch',
                    {'for_quant+++slide1+++acq': np.ones((200, 200, 3))},
                                                mcd_filepath)
    assert np.sum(new_stitch['test_stitch']) == 0
    new_stitch = stitch_mcd_blends_from_gallery(new_stitch, 'test_stitch',
                                                {'test+++slide1+++chr10-h54h54-Gd158_2_18': np.ones((200, 200, 3))},
                                                mcd_filepath)
    assert np.sum(new_stitch['test_stitch']) > 0

    # when two MCD slides mismatch
    mcd_2_w, mcd_2_h = MCDAcqCoordinateParser([os.path.join(get_current_dir, 'query.mcd')],
                                'query+++slide1+++PAP_1').get_roi_slide_boundary_point()
    assert not cur_roi_slide_matches_stitch(slide_height, slide_width, mcd_2_h, mcd_2_w)
    assert not cur_roi_slide_matches_stitch(slide_height, slide_width, None, None)
