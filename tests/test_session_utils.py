import os
from ccramic.utils.session import remove_ccramic_caches

def test_session_cache_clearing():
    unique_id = "dsfpihdsfpidjfd"
    if not os.path.isdir(os.path.join('/tmp/', unique_id)):
        os.mkdir(os.path.join('/tmp/', unique_id))
    if not os.path.isdir(os.path.join('/tmp/', unique_id, 'ccramic_cache')):
        os.mkdir(os.path.join('/tmp/', unique_id, 'ccramic_cache'))
    if not os.path.isdir(os.path.join('/tmp/', unique_id, 'different_directory')):
        os.mkdir(os.path.join('/tmp/', unique_id, 'different_directory'))
    assert os.path.isdir(os.path.join('/tmp/', unique_id, 'ccramic_cache'))
    assert os.path.isdir(os.path.join('/tmp/', unique_id, 'different_directory'))
    remove_ccramic_caches('/tmp/')
    assert not os.path.isdir(os.path.join('/tmp/', unique_id, 'ccramic_cache'))
    # assert os.path.isdir(os.path.join('/tmp/', unique_id, 'different_directory'))
