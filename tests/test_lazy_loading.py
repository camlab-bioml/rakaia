import os
import numpy as np
from rakaia.parsers.pixel import (
    FileParser,
    roi_requires_single_marker_load)
from rakaia.parsers.lazy_load import (
    SingleMarkerLazyLoader,
    parse_files_for_lazy_loading)

def test_requirements_single_marker_lazy_load():
    array = np.zeros((5000, 5000))
    assert roi_requires_single_marker_load(array, 50)
    assert roi_requires_single_marker_load(35000000, 55)
    assert not roi_requires_single_marker_load(array, 1)
    assert not roi_requires_single_marker_load(np.zeros((1000, 1000)), 1000)
    assert not roi_requires_single_marker_load(10000, 100)

def test_lazy_loading_single_marker_mcd(get_current_dir):
    parser = FileParser([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = parser.image_dict
    session_config = {"uploads": [os.path.join(get_current_dir, "query.mcd")]}
    assert parse_files_for_lazy_loading(session_config, 'query+++slide0+++Xylene_5')
    selection = ['Ir191']
    single_load = SingleMarkerLazyLoader(uploaded_dict, 'query+++slide0+++Xylene_5',
                                         session_config, selection).get_image_dict()
    assert all(single_load['query+++slide0+++Xylene_5'][elem] is None for elem in
               single_load['query+++slide0+++Xylene_5'].keys() if elem not in selection)
    assert all(isinstance(single_load['query+++slide0+++Xylene_5'][elem], np.ndarray) for elem in
               single_load['query+++slide0+++Xylene_5'].keys() if elem in selection)


def test_lazy_loading_single_marker_spatial(get_current_dir):
    session_config = {"uploads": [os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    assert parse_files_for_lazy_loading(session_config, 'visium_thalamus+++slide0+++acq')
    visium_parser = FileParser(session_config['uploads'])
    assert all(elem is None for elem in visium_parser.image_dict['visium_thalamus+++slide0+++acq'].values())
    lazy_load = SingleMarkerLazyLoader(visium_parser.image_dict, 'visium_thalamus+++slide0+++acq',
                    session_config, 'Sox17')
    single_load = lazy_load.get_image_dict()
    dims = lazy_load.get_region_dim()
    assert all(single_load['visium_thalamus+++slide0+++acq'][elem] is None for elem in
               single_load['visium_thalamus+++slide0+++acq'].keys() if elem != 'Sox17')
    assert all(isinstance(single_load['visium_thalamus+++slide0+++acq'][elem], np.ndarray) for elem in
               single_load['visium_thalamus+++slide0+++acq'].keys() if elem == 'Sox17')
    assert dims == (1011, 925, 1582, 1691)

def test_lazy_loading_single_marker_tiff(get_current_dir):
    uploads = {'uploads': [os.path.join(get_current_dir, "for_recolour.tiff")]}
    uploaded_dict = FileParser(uploads['uploads']).image_dict
    assert all(uploaded_dict['for_recolour+++slide0+++acq'][elem] is None for elem in
               uploaded_dict['for_recolour+++slide0+++acq'].keys())
    lazy_loader = SingleMarkerLazyLoader(uploaded_dict, 'for_recolour+++slide0+++acq',
                                         uploads, 'channel_1')
    single_load = lazy_loader.get_image_dict()
    dims = lazy_loader.get_region_dim()
    assert all(isinstance(single_load['for_recolour+++slide0+++acq'][elem], np.ndarray) for elem in
               single_load['for_recolour+++slide0+++acq'].keys())
    assert dims == (600, 600, 0, 0)
