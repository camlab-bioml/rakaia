import pytest
from ccramic.parsers.pixel_level_parsers import *
from scipy.sparse import csr_matrix
import numpy as np

def test_basic_parser_tiff_to_dict(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "for_recolour.tiff")])
    assert isinstance(uploaded, tuple)
    uploaded_dict = uploaded[0]
    assert len(uploaded_dict['metadata']) > 0
    assert 'for_recolour+++slide0+++acq0' in uploaded_dict.keys()
    assert not 'experiment1' in uploaded_dict.keys()
    assert len(uploaded_dict['for_recolour+++slide0+++acq0'].keys()) == 1

    blending_dict = create_new_blending_dict(uploaded_dict)
    assert all([elem in ['#FFFFFF', None] for elem in \
                blending_dict['channel_1'].values()])

def test_basic_parser_fake_mcd(get_current_dir):
    with pytest.raises(ValueError):
        populate_upload_dict([os.path.join(get_current_dir, "fake_dataset.mcd")])


def test_basic_parser_from_mcd(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = uploaded[0]
    assert 'query+++slide0+++Xylene' in uploaded_dict.keys()
    assert 'metadata' in uploaded_dict.keys()
    assert len(uploaded_dict['query+++slide0+++Xylene']) == 11
    # the values will all be none for the mcd because of lazy loadinfg
    assert all([value is None for value in uploaded_dict['query+++slide0+++Xylene'].values()])
    dataset_info = uploaded[-1]
    assert len(dataset_info['ROI']) == 6
    assert '11 markers' in dataset_info['Panel']

def test_basic_parser_blend_dict_from_lazy_loading(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = uploaded[0]
    assert 'query+++slide0+++Xylene' in uploaded_dict.keys()
    assert all([elem is None for elem in uploaded_dict['query+++slide0+++Xylene'].values()])
    session_config = {"uploads": [os.path.join(get_current_dir, "query.mcd")]}
    new_upload_dict = populate_upload_dict_by_roi(uploaded_dict, 'query+++slide0+++Xylene', session_config)
    assert all([elem is not None for elem in new_upload_dict['query+++slide0+++Xylene'].values()])

def test_basic_parser_from_text(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "query_from_text.txt")])
    uploaded_dict = uploaded[0]
    assert 'metadata' in uploaded_dict.keys()
    assert 'query_from_text+++slide0+++0' in uploaded_dict.keys()
    assert len(uploaded_dict['query_from_text+++slide0+++0'].keys()) == 4

def test_basic_parser_from_h5py(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "data.h5")])
    uploaded_dict = uploaded[0]
    assert 'metadata' in uploaded_dict.keys()
    assert 'cycA1A2_Ir_TBS_3_8+++slide0+++0' in uploaded_dict.keys()
    assert len(uploaded_dict['cycA1A2_Ir_TBS_3_8+++slide0+++0'].keys()) == 4

def test_identify_sparse_matrices():
    array = np.full((700, 700), 3)
    array_return = dense_array_to_sparse(array)
    assert np.array_equal(array, array_return)
    sparse = csr_matrix(array)
    array_return = dense_array_to_sparse(sparse)
    assert not np.array_equal(sparse, array_return)
