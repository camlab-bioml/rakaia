import os
import time
import numpy as np
import pytest
from rakaia.utils.session import (
    remove_rakaia_caches,
    non_truthy_to_prevent_update,
    validate_session_upload_config,
    channel_dropdown_selection,
    sleep_on_small_roi,
    set_data_selection_after_import,
    roi_from_anndata_file)
import tempfile
from dash.exceptions import PreventUpdate

def test_session_cache_clearing():
    unique_id = "dsfpihdsfpidjfd"
    with tempfile.TemporaryDirectory() as tmpdirname:
        if not os.path.isdir(os.path.join(tmpdirname, unique_id)):
            os.mkdir(os.path.join(tmpdirname, unique_id))
        if not os.path.isdir(os.path.join(tmpdirname, unique_id, 'rakaia_cache')):
            os.mkdir(os.path.join(tmpdirname, unique_id, 'rakaia_cache'))
        if not os.path.isdir(os.path.join(tmpdirname, unique_id, 'different_directory')):
            os.mkdir(os.path.join(tmpdirname, unique_id, 'different_directory'))
        assert os.path.isdir(os.path.join(tmpdirname, unique_id, 'rakaia_cache'))
        assert os.path.isdir(os.path.join(tmpdirname, unique_id, 'different_directory'))
        remove_rakaia_caches(tmpdirname)
        assert not os.path.isdir(os.path.join(tmpdirname, unique_id, 'rakaia_cache'))

def test_non_truthy_to_update_prevention():
    real_string = "this is real"
    assert non_truthy_to_prevent_update(real_string) == real_string
    assert non_truthy_to_prevent_update(True)
    with pytest.raises(PreventUpdate):
        non_truthy_to_prevent_update([])
    with pytest.raises(PreventUpdate):
        non_truthy_to_prevent_update(False)
    with pytest.raises(PreventUpdate):
        non_truthy_to_prevent_update("")

def test_validate_session_uploads():
    assert validate_session_upload_config() == {'uploads': []}
    assert validate_session_upload_config({"fake_key": "fake_val"}) == {'uploads': []}
    upload_dict = {'uploads': ['file_1.mcd']}
    assert validate_session_upload_config(upload_dict) == upload_dict

def test_generate_channel_dropdowns():
    names = {'ArAr80': 'Gas_1', 'Y89': 'Gas_2', 'In113': 'Histone'}
    channels = {'ArAr80': '80ArAr', 'Y89': 'Gas', 'In113': 'Histone_126((2979))In113'}
    options = channel_dropdown_selection(channels, names)
    assert options == [{'label': 'Gas_1', 'value': 'ArAr80'},
                       {'label': 'Gas_2', 'value': 'Y89'},
                       {'label': 'Histone', 'value': 'In113'}]
    assert not channel_dropdown_selection(None, names)
    assert not channel_dropdown_selection({}, {})

def test_roi_pause():
    small_roi = np.zeros((350, 350))
    start = time.time()
    sleep_on_small_roi(small_roi.shape)
    end = time.time()
    assert (end - start) >= 2.00
    larger_roi = np.zeros((401, 401))
    start = time.time()
    sleep_on_small_roi(larger_roi.shape)
    end = time.time()
    assert (end - start) < 2.00

def test_set_data_selection():
    roi_options = ["roi_1", "roi_2"]
    assert set_data_selection_after_import(roi_options, "roi_2") == "roi_2"
    assert set_data_selection_after_import(roi_options, None) == "roi_1"
    assert not set_data_selection_after_import(None, None)
    assert not set_data_selection_after_import(None, "roi_2")

def test_roi_from_anndata():
    uploads = {"uploads": ['not_anndata.txt', 'not_anndata.mcd', 'from_anndata.h5ad']}
    assert roi_from_anndata_file(uploads, "from_anndata+++slideNA+++acq")
    assert not roi_from_anndata_file(uploads, "not_anndata+++slideNA+++acq")
