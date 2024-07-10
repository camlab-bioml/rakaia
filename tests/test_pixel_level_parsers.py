import pandas as pd
import pytest
from rakaia.parsers.pixel import (
    FileParser,
    create_new_blending_dict,
    populate_image_dict_from_lazy_load,
    sparse_array_to_dense,
    convert_between_dense_sparse_array,
    convert_rgb_to_greyscale,
    populate_alias_dict_from_editable_metadata,
    check_blend_dictionary_for_blank_bounds_by_channel,
    check_empty_missing_layer_dict,
    NoAcquisitionsParsedError)
from scipy.sparse import csc_matrix
import numpy as np
from rakaia.utils.alert import PanelMismatchError
import os
from scipy.sparse import issparse

def test_basic_conversion_rgb_to_greyscale():
    rgb = np.full((1000, 1000, 3), 255)
    greyscale = convert_rgb_to_greyscale(rgb)
    assert len(greyscale.shape) == 2
    greyscale_2 = convert_rgb_to_greyscale(greyscale)
    assert np.array_equal(greyscale, greyscale_2)


def test_basic_parser_tiff_to_dict(get_current_dir):
    uploaded_dict = FileParser([os.path.join(get_current_dir, "for_recolour.tiff")]).image_dict
    assert len(uploaded_dict['metadata']) > 0
    assert 'for_recolour+++slide0+++acq0' in uploaded_dict.keys()
    assert not 'experiment1' in uploaded_dict.keys()
    assert len(uploaded_dict['for_recolour+++slide0+++acq0'].keys()) == 1
    assert all([elem is None for elem in uploaded_dict['for_recolour+++slide0+++acq0'].values()])

    blending_dict = create_new_blending_dict(uploaded_dict)
    assert all([elem in ['#FFFFFF', None] for elem in \
                blending_dict['channel_1'].values()])

    with pytest.raises(TypeError):
        FileParser([os.path.join(get_current_dir, "for_recolour.tiff")],
                   array_store_type="fake")

def test_basic_parser_fake_mcd(get_current_dir):
    with pytest.raises(ValueError):
        FileParser([os.path.join(get_current_dir, "fake_dataset.mcd")])


def test_basic_parser_from_mcd(get_current_dir):
    parser = FileParser([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = parser.image_dict
    assert 'query+++slide0+++Xylene_5' in uploaded_dict.keys()
    assert 'metadata' in uploaded_dict.keys()
    assert len(uploaded_dict['query+++slide0+++Xylene_5']) == 11
    # the values will all be none for the mcd because of lazy loading
    assert all([value is None for value in uploaded_dict['query+++slide0+++Xylene_5'].values()])
    dataset_info = parser.get_parsed_information()
    assert len(dataset_info['ROI']) == 6
    assert '11 markers' in dataset_info['Panel']

def test_empty_information_parser(get_current_dir):
    parser = FileParser([os.path.join(get_current_dir, "empty.txt")])
    with pytest.raises(NoAcquisitionsParsedError):
        parser.get_parsed_information()

def test_basic_parser_exceptions(get_current_dir):
    """
    Exceptions on different panel lengths or invalid filepaths
    """
    with pytest.raises(PanelMismatchError):
        FileParser([os.path.join(get_current_dir, "query.mcd"),
                         os.path.join(get_current_dir, "for_recolour.tiff")])
        FileParser([os.path.join(get_current_dir, "query.mcd"),
                    os.path.join(get_current_dir, "query_from_text.txt")])
    with pytest.raises(TypeError):
        FileParser([os.path.join(get_current_dir, "point_annotations.csv")])

def test_basic_parser_blend_dict_from_lazy_loading(get_current_dir):
    parser = FileParser([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = parser.image_dict
    assert 'query+++slide0+++Xylene_5' in uploaded_dict.keys()
    assert all([elem is None for elem in uploaded_dict['query+++slide0+++Xylene_5'].values()])
    session_config = {"uploads": [os.path.join(get_current_dir, "query.mcd")]}
    new_upload_dict = populate_image_dict_from_lazy_load(uploaded_dict, 'query+++slide0+++Xylene_5', session_config)
    assert all([elem is not None for elem in new_upload_dict['query+++slide0+++Xylene_5'].values()])

def test_basic_parser_lazy_loading_2(get_current_dir):
    uploaded = FileParser([os.path.join(get_current_dir, "query_from_text.txt")])
    uploaded_dict = uploaded.image_dict
    assert all([elem is None for elem in uploaded_dict['query_from_text+++slide0+++0'].values()])
    session_config = {"uploads": [os.path.join(get_current_dir, "query_from_text.txt")]}
    new_upload_dict = populate_image_dict_from_lazy_load(uploaded_dict, 'query_from_text+++slide0+++0', session_config)
    assert all([elem is not None for elem in new_upload_dict['query_from_text+++slide0+++0'].values()])



def test_basic_parser_from_text(get_current_dir):
    uploaded = FileParser([os.path.join(get_current_dir, "query_from_text.txt")])
    uploaded_dict = uploaded.image_dict
    assert 'metadata' in uploaded_dict.keys()
    assert 'query_from_text+++slide0+++0' in uploaded_dict.keys()
    assert len(uploaded_dict['query_from_text+++slide0+++0'].keys()) == 4
    assert all([elem is None for elem in uploaded_dict['query_from_text+++slide0+++0'].values()])

def test_basic_parser_from_h5py(get_current_dir):
    uploaded = FileParser([os.path.join(get_current_dir, "data.h5")])
    uploaded_dict = uploaded.image_dict
    assert 'metadata' in uploaded_dict.keys()
    assert isinstance(uploaded_dict['metadata'], pd.DataFrame)
    assert 'test---slide0---chr10-h54h54-Gd158_2_18' in uploaded_dict.keys()
    assert len(uploaded_dict['test---slide0---chr10-h54h54-Gd158_2_18'].keys()) == 12 == len(uploaded_dict['metadata'])

def test_identify_sparse_matrices():
    array = np.full((700, 700), 3)
    array_return = sparse_array_to_dense(array)
    assert np.array_equal(array, array_return)
    sparse = csc_matrix(array)
    array_return = sparse_array_to_dense(sparse)
    assert not np.array_equal(sparse, array_return)

def test_conversion_between_sparse_dense_matrices():
    array = np.full((700, 700), 3)
    sparse = convert_between_dense_sparse_array(array, "sparse")
    assert issparse(sparse)
    array_back = convert_between_dense_sparse_array(sparse, "dense")
    assert not issparse(array_back)
    sparse_to_sparse = convert_between_dense_sparse_array(sparse, "sparse")
    assert issparse(sparse_to_sparse)
    dense_to_dense = convert_between_dense_sparse_array(array, "dense")
    assert not issparse(dense_to_dense)

    with pytest.raises(TypeError):
        convert_between_dense_sparse_array(array, "fake_type")


def test_basic_metadata_alias_parse():
    editable_metadata = [
        {'Channel Order': 1, 'Channel Name': 'channel_1', 'Channel Label': 'channel_1', 'rakaia Label': 'FSP1'},
        {'Channel Order': 2, 'Channel Name': 'channel_2', 'Channel Label': 'channel_2', 'rakaia Label': 'SMA'},
        {'Channel Order': 3, 'Channel Name': 'channel_3', 'Channel Label': 'channel_3', 'rakaia Label': 'H3K27me3'},
        {'Channel Order': 4, 'Channel Name': 'channel_4', 'Channel Label': 'channel_4', 'rakaia Label': 'pan_CK'},
        {'Channel Order': 5, 'Channel Name': 'channel_5', 'Channel Label': 'channel_5', 'rakaia Label': 'Fibronectin'},
        {'Channel Order': 6, 'Channel Name': 'channel_6', 'Channel Label': 'channel_6', 'rakaia Label': 'CD44'}]
    labels = populate_alias_dict_from_editable_metadata(editable_metadata)
    assert len(labels) == 6
    assert labels['channel_4'] == 'pan_CK'

    bad_meta = [
        {'Channel Order': 1, 'Channel Name': 'channel_1', 'Channel Label': 'channel_1', 'rakaia Label': 'FSP1'},
        {'Channel Order': 2, 'Channel Label': 'channel_2', 'rakaia Label': 'SMA'}]
    labels = populate_alias_dict_from_editable_metadata(bad_meta)
    assert len(labels) == 1

    bad_meta_2 = [{'Channel Order': 1, 'Channel Name': 'channel_1', 'Channel Label': 'channel_1', 'rakaia Label': 'FSP1'},
        {'Channel Order': 2, 'Channel Name': 'channel_2', 'Channel Label': 'channel_2'}]
    labels = populate_alias_dict_from_editable_metadata(bad_meta_2)
    assert len(labels) == 2
    assert labels['channel_2'] == 'channel_2'

def test_check_empty_blend_bounds():
    blend_dict = {"channel_1": {"x_lower_bound": None, "x_upper_bound": None}}
    channel_dict = {"roi_1": {"channel_1": np.full((1000, 1000), 7)}}
    blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(blend_dict, "channel_1", channel_dict, "roi_1")
    assert blend_dict == {'channel_1': {'x_lower_bound': 0, 'x_upper_bound': 7.0}}
    blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(blend_dict, "channel_1", channel_dict, "roi_1")
    assert blend_dict == {'channel_1': {'x_lower_bound': 0, 'x_upper_bound': 7.0}}

def test_layer_dict_status():
    assert check_empty_missing_layer_dict(None, "roi_1") == {'roi_1': {}}
    assert check_empty_missing_layer_dict({'roi_1': {}}, "roi_2") == {'roi_2': {}}
    assert check_empty_missing_layer_dict({'roi_2': {"channel_1": 1}}, "roi_2") == {'roi_2': {"channel_1": 1}}
