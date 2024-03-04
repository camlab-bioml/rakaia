import os

import pytest

from ccramic.utils.session import (
    remove_ccramic_caches,
    non_truthy_to_prevent_update,
    validate_session_upload_config)
import tempfile
from dash.exceptions import PreventUpdate

def test_session_cache_clearing():
    unique_id = "dsfpihdsfpidjfd"
    with tempfile.TemporaryDirectory() as tmpdirname:
        if not os.path.isdir(os.path.join(tmpdirname, unique_id)):
            os.mkdir(os.path.join(tmpdirname, unique_id))
        if not os.path.isdir(os.path.join(tmpdirname, unique_id, 'ccramic_cache')):
            os.mkdir(os.path.join(tmpdirname, unique_id, 'ccramic_cache'))
        if not os.path.isdir(os.path.join(tmpdirname, unique_id, 'different_directory')):
            os.mkdir(os.path.join(tmpdirname, unique_id, 'different_directory'))
        assert os.path.isdir(os.path.join(tmpdirname, unique_id, 'ccramic_cache'))
        assert os.path.isdir(os.path.join(tmpdirname, unique_id, 'different_directory'))
        remove_ccramic_caches(tmpdirname)
        assert not os.path.isdir(os.path.join(tmpdirname, unique_id, 'ccramic_cache'))
        # assert os.path.isdir(os.path.join(tmpdirname, unique_id, 'different_directory'))

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
