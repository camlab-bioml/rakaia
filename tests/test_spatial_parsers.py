import os
import tempfile
import numpy as np
import anndata as ad
import pytest
from rakaia.parsers.pixel import FileParser
from rakaia.parsers.lazy_load import parse_files_for_lazy_loading
from rakaia.parsers.spatial import (
    spatial_canvas_dimensions,
    spatial_grid_single_marker,
    check_spatial_array_multi_channel,
    get_spatial_spot_radius,
    detect_spatial_capture_size,
    visium_has_scaling_factors,
    is_spatial_dataset,
    spatial_selection_can_transfer_coordinates,
    visium_coords_to_wsi_from_zoom,
    get_visium_bin_scaling, xenium_coords_to_wsi_from_zoom, is_zarr_store, ZarrSDParser)
from rakaia.parsers.object import visium_mask

def test_identify_h5ad_in_uploads(get_current_dir):
    uploads = {"uploads": [os.path.join(get_current_dir, "for_recolour.tiff"),
                           os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    h5ad_found = parse_files_for_lazy_loading(uploads, "visium_thalamus+++slideNA+++acq")
    assert h5ad_found == os.path.join(get_current_dir, "visium_thalamus.h5ad")
    assert parse_files_for_lazy_loading(uploads, "other_visium+++slideNA+++acq") is None
    assert parse_files_for_lazy_loading([], "visium_thalamus+++slideNA+++acq") is None

def test_basic_visium_anndata_parser(get_current_dir):
    visium_parser = FileParser([os.path.join(get_current_dir, "visium_thalamus.h5ad")])
    assert len(visium_parser.metadata_labels) == len(visium_parser.metadata_channels) == 250
    assert all(elem is None for elem in visium_parser.image_dict['visium_thalamus+++slideNA+++acq'].values())

def test_visium_generate_spot_grid(get_current_dir):
    grid_image = spatial_grid_single_marker(os.path.join(get_current_dir, "visium_thalamus.h5ad"),
                                                'Sox17')
    assert grid_image[461, 680] == np.max(grid_image)
    assert grid_image[10, 10] == 0

    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(
        os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    assert grid_image.shape == (grid_height, grid_width) == (925, 1011)

    adata = ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    assert get_spatial_spot_radius(adata, 3) == 89
    no_spatial = ad.AnnData(X=adata.X)
    no_spatial.var_names = adata.var_names
    assert all(elem is None for elem in spatial_canvas_dimensions(no_spatial))
    # if no radius is found, use the default of 1
    assert get_spatial_spot_radius(no_spatial) == 1
    assert get_spatial_spot_radius(no_spatial, 3) == 3
    assert detect_spatial_capture_size() == 65
    assert not visium_has_scaling_factors(no_spatial)

    with pytest.raises(ValueError):
        spatial_grid_single_marker(os.path.join(get_current_dir, "visium_thalamus.h5ad"),
                                       'fake_gene')
    with pytest.raises(ValueError):
        spatial_grid_single_marker(no_spatial, 'Sox17')

def test_parse_image_dict_for_missing_spot_grids(get_current_dir):
    adata = ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))
    assert is_spatial_dataset(adata)
    image_dict = {'visium_thalamus+++slideNA+++acq': {marker: None for marker in list(adata.var_names)}}
    image_dict_back = check_spatial_array_multi_channel(image_dict,
            'visium_thalamus+++slideNA+++acq', adata, list(adata.var_names)[0:2])
    for marker in image_dict_back['visium_thalamus+++slideNA+++acq'].keys():
        if marker in list(adata.var_names)[0:2]:
            assert image_dict_back['visium_thalamus+++slideNA+++acq'][marker] is not None
            assert image_dict_back['visium_thalamus+++slideNA+++acq'][marker].shape == (925, 1011)
        else:
            assert image_dict_back['visium_thalamus+++slideNA+++acq'][marker] is None


def test_parse_visium_spot_mask(get_current_dir):
    uploads = {"uploads": [os.path.join(get_current_dir, "visium_thalamus.h5ad")]}
    mask_dict, names = visium_mask({}, 'visium_thalamus---slideNA---acq',
                                   uploads, delimiter="---")
    assert 'visium_thalamus' in names
    assert 'visium_thalamus' in mask_dict.value.keys()
    assert mask_dict.value['visium_thalamus']['raw'].shape == (925, 1011)
    assert (np.max(mask_dict.value['visium_thalamus']['raw']) ==
            len(ad.read_h5ad(os.path.join(get_current_dir, "visium_thalamus.h5ad"))))

def test_detect_spatial_can_perform_coord_transfer(get_current_dir):
    """
    Currently, only visium spot-based can perform spatial coordinate transfer
    """
    uploads = {"uploads": ['fake_file.txt', 'data.h5', 'mask.tiff',
                           os.path.join(get_current_dir, 'visium_thalamus.h5ad')]}
    can_transfer, file = spatial_selection_can_transfer_coordinates('visium_thalamus+++slideNA+++acq', uploads)
    assert can_transfer
    assert str(file) == str(os.path.join(get_current_dir, 'visium_thalamus.h5ad'))
    can_transfer, file = spatial_selection_can_transfer_coordinates('fake_file+++slideNA+++acq', uploads)
    assert not can_transfer
    assert file is None

def test_visium_spot_coords_to_wsi(get_current_dir):
    bounds = {'xaxis.range[0]': 283.4, 'xaxis.range[1]': 741.5,
              'yaxis.range[0]':  683.8, 'yaxis.range[1]': 264.6}
    adata = ad.read_h5ad(os.path.join(get_current_dir, 'visium_thalamus.h5ad'))
    string_coords = visium_coords_to_wsi_from_zoom(bounds, adata)
    x, y, width, height = tuple([float(elem) for elem in string_coords.split(",")])
    assert width > height
    assert y > x
    x_min, y_min = np.min((adata.obsm['spatial']), axis=0)
    x_max, y_max = np.max((adata.obsm['spatial']), axis=0)
    assert y_min < y < y_max
    assert x_min < x < x_max

def test_hd_visium_spot_coords_to_wsi(get_current_dir):
    bounds = {'xaxis.range[0]': 30.6, 'xaxis.range[1]': 59.6,
              'yaxis.range[0]':  27.8, 'yaxis.range[1]': 57.8}
    adata = ad.read_h5ad(os.path.join(get_current_dir, 'intestine_hd_subset.h5ad'))
    string_coords = visium_coords_to_wsi_from_zoom(bounds, adata)
    x, y, width, height = tuple([float(elem) for elem in string_coords.split(",")])
    bin_size = get_visium_bin_scaling(adata)
    assert x > y
    x_min, y_min = np.min((adata.obsm['spatial']) * bin_size , axis=0)
    x_max, y_max = np.max((adata.obsm['spatial']) * bin_size, axis=0)
    assert y_min < y < y_max
    assert x_min < x < x_max

def test_xenium_coords_to_wsi(get_current_dir):
    bounds = {'xaxis.range[0]': 36.7, 'xaxis.range[1]': 281.0,
              'yaxis.range[0]': 31.8, 'yaxis.range[1]': 147.1}
    string_coords = xenium_coords_to_wsi_from_zoom(bounds,
                os.path.join(get_current_dir, 'melanoma_xenium_subset.h5ad'),
                os.path.join(get_current_dir, 'melanoma_xenium_transformation.csv'))
    x, y, width, height = tuple([float(elem) for elem in string_coords.split(",")])
    assert y > x
    assert height > width


def test_is_zarr_store(get_current_dir):
    assert not is_zarr_store(os.path.join(get_current_dir, 'wsi'))
    assert is_zarr_store(os.path.join(get_current_dir, 'subset_visium.zarr'))

def test_parse_sd_visium(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        parsed = ZarrSDParser(os.path.join(get_current_dir, 'subset_visium.zarr'),
                              tmpdirname).get_files()[0]
        assert len(parsed['uploads']) == 2
        assert all('ST8059' in elem for elem in parsed['uploads'])
        assert all (os.path.exists(elem) for elem in parsed['uploads'])
        for path in parsed['uploads']:
            if os.access(path, os.W_OK):
                os.remove(path)
        # if current uploads already exist
        parsed = ZarrSDParser(os.path.join(get_current_dir, 'subset_visium.zarr'),
                tmpdirname, {'uploads': ['already_there.h5ad']}).get_files()[0]
        assert len(parsed['uploads']) == 3
        for path in parsed['uploads']:
            if os.path.isfile(path) and os.access(path, os.W_OK):
                os.remove(path)

def test_parse_sd_xenium(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        parsed = ZarrSDParser(os.path.join(get_current_dir, 'subset_xenium.zarr'),
                              tmpdirname, None, {
                'mask_1': 'fake_path_to_mask.tiff'}).get_files()
        files = parsed[0]
        masks = parsed[2]
        assert len(files['uploads']) == 1
        # count the existing session mask when adding new cell segmentation mask
        assert len(masks) == 2
        for file in (files['uploads'][0], masks['subset_xenium_zarr']):
            if os.access(file, os.W_OK):
                os.remove(file)

def test_parse_sd_visium_hd(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        parsed = ZarrSDParser(os.path.join(get_current_dir, 'subset_visium_hd.zarr'),
                              os.path.join(tmpdirname, 'other_spatial')).get_files()[0]
        assert len(parsed['uploads']) == 3
        for bin_size in ['002um', '008um', '016um']:
            assert any(bin_size in file_out for file_out in parsed['uploads'])
        for path in parsed['uploads']:
            if os.access(path, os.W_OK):
                os.remove(path)

def test_parse_sd_other_spatial(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # if not detected as 10x, export each table as a spatial region with the table key in the name
        parsed = ZarrSDParser(os.path.join(get_current_dir, 'subset_other_spatial.zarr'),
                                  tmpdirname).get_files()[0]
        assert len(parsed['uploads']) == 1
        assert 'subset_other_spatial_zarr_table' in parsed['uploads'][0]
        for path in parsed['uploads']:
            if os.access(path, os.W_OK):
                os.remove(path)
