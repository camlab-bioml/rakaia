import os
import numpy as np
from rakaia.parsers.pixel import (
    FileParser,
    roi_requires_single_marker_load)
from rakaia.parsers.lazy_load import SingleMarkerLazyLoader

def test_requirements_single_marker_lazy_load():
    array = np.zeros((5000, 5000))
    assert roi_requires_single_marker_load(array, 50)
    assert not roi_requires_single_marker_load(array, 1)
    assert not roi_requires_single_marker_load(np.zeros((1000, 1000)), 1000)

def test_lazy_loading_single_marker_mcd(get_current_dir):
    parser = FileParser([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = parser.image_dict
    session_config = {"uploads": [os.path.join(get_current_dir, "query.mcd")]}
    selection = ['Ir191']
    single_load = SingleMarkerLazyLoader(uploaded_dict, 'query+++slide0+++Xylene_5',
                                         session_config, selection).get_image_dict()
    assert all(single_load['query+++slide0+++Xylene_5'][elem] is None for elem in
               single_load['query+++slide0+++Xylene_5'].keys() if elem not in selection)
    assert all(isinstance(single_load['query+++slide0+++Xylene_5'][elem], np.ndarray) for elem in
               single_load['query+++slide0+++Xylene_5'].keys() if elem in selection)


def test_lazy_loading_single_marker_spatial(get_current_dir):
    session_config = {"uploads": [os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    visium_parser = FileParser(session_config['uploads'])
    assert all(elem is None for elem in visium_parser.image_dict['visium_thalamus+++slide0+++acq'].values())
    single_load = SingleMarkerLazyLoader(visium_parser.image_dict, 'visium_thalamus+++slide0+++acq',
                    session_config, 'Sox17').get_image_dict()
    assert all(single_load['visium_thalamus+++slide0+++acq'][elem] is None for elem in
               single_load['visium_thalamus+++slide0+++acq'].keys() if elem != 'Sox17')
    assert all(isinstance(single_load['visium_thalamus+++slide0+++acq'][elem], np.ndarray) for elem in
               single_load['visium_thalamus+++slide0+++acq'].keys() if elem == 'Sox17')
    dims = SingleMarkerLazyLoader(visium_parser.image_dict, 'visium_thalamus+++slide0+++acq',
                                         session_config, 'Sox17').get_region_dim()

    assert dims == (1011, 925, 1582, 1691)
