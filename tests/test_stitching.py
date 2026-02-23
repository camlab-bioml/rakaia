import tempfile
import os
import numpy as np
from rakaia.stitch import (
    update_stitch_cache_with_blend,
    stitch_cache_dropdown_labels,
    download_stitch_image)

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
