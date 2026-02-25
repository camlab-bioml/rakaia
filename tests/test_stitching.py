import tempfile
import os
import numpy as np
import PIL
import pytest
from rakaia.stitch import (
    update_stitch_cache_with_blend,
    stitch_cache_dropdown_labels,
    download_stitch_image,
    stitch_image_preview)
from rakaia.stitch.mcd import MCDAcqCoordinateParser

def test_stitch_cache_update(stitch_cache):
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
