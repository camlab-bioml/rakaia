import os
import numpy as np
import anndata as ad
import pytest
from rakaia.parsers.pixel import (
    FileParser, parse_files_for_h5ad)
from rakaia.parsers.spatial import (
    spatial_canvas_dimensions,
    spatial_grid_single_marker,
    check_spot_grid_multi_channel,
    get_spatial_spot_radius,
    detect_spatial_capture_size,
    visium_has_scaling_factors)
from rakaia.parsers.object import visium_mask

def test_identify_h5ad_in_uploads(get_current_dir):
    uploads = {"uploads": [os.path.join(get_current_dir, "for_recolour.tiff"),
                           os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    h5ad_found = parse_files_for_h5ad(uploads, "visium_thalamus+++slide0+++acq")
    assert h5ad_found == os.path.join(get_current_dir, "visium_thalamus.h5ad")
    assert parse_files_for_h5ad(uploads, "other_visium+++slide0+++acq") is None
    assert parse_files_for_h5ad([], "visium_thalamus+++slide0+++acq") is None

def test_basic_visium_anndata_parser(get_current_dir):
    visium_parser = FileParser([os.path.join(get_current_dir, "visium_thalamus.h5ad")])
    assert len(visium_parser.metadata_labels) == len(visium_parser.metadata_channels) == 250
    assert all(elem is None for elem in visium_parser.image_dict['visium_thalamus+++slide0+++acq'].values())


def test_visium_generate_spot_grid(get_current_dir):
    grid_image = spatial_grid_single_marker(os.path.join(get_current_dir, "visium_thalamus.h5ad"),
                                                'Sox17')
    assert grid_image[481, 700] == np.max(grid_image)
    assert grid_image[10, 10] == 0

    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(
        os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    assert grid_image.shape == (grid_height, grid_width) == (961, 1051)

    adata = ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    assert get_spatial_spot_radius(adata) == 89
    no_spatial = ad.AnnData(X=adata.X)
    no_spatial.var_names = adata.var_names
    assert all(elem is None for elem in spatial_canvas_dimensions(no_spatial))
    # if no radius is found, use the default of 1
    assert get_spatial_spot_radius(no_spatial) == 1
    assert detect_spatial_capture_size(no_spatial) == 65
    assert not visium_has_scaling_factors(no_spatial)

    with pytest.raises(ValueError):
        spatial_grid_single_marker(os.path.join(get_current_dir, "visium_thalamus.h5ad"),
                                       'fake_gene')
    with pytest.raises(ValueError):
        spatial_grid_single_marker(no_spatial, 'Sox17')

def test_parse_image_dict_for_missing_spot_grids(get_current_dir):
    adata = ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    image_dict = {'visium_thalamus+++slide0+++acq': {marker: None for marker in list(adata.var_names)}}
    image_dict_back = check_spot_grid_multi_channel(image_dict,
            'visium_thalamus+++slide0+++acq', adata, list(adata.var_names)[0:2])
    for marker in image_dict_back['visium_thalamus+++slide0+++acq'].keys():
        if marker in list(adata.var_names)[0:2]:
            assert image_dict_back['visium_thalamus+++slide0+++acq'][marker] is not None
            assert image_dict_back['visium_thalamus+++slide0+++acq'][marker].shape == (961, 1051)
        else:
            assert image_dict_back['visium_thalamus+++slide0+++acq'][marker] is None


def test_parse_visium_spot_mask(get_current_dir):
    uploads = {"uploads": [os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    mask_dict, names = visium_mask({}, 'visium_thalamus+++slide0+++acq',
                                   uploads)
    assert 'visium_thalamus' in names
    assert 'visium_thalamus' in mask_dict.value.keys()
    assert mask_dict.value['visium_thalamus']['raw'].shape == (961, 1051)
    assert (np.max(mask_dict.value['visium_thalamus']['raw']) ==
            len(ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))))
