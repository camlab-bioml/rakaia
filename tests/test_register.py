import tempfile
import xml.etree.ElementTree as ET
import os
import platform
import dash
import numpy as np
import pytest
import shutil
import tifffile
from rakaia.register.process import (
    dzi_tiles_from_image_path,
    update_wsi_hash,
    wsi_from_local_path,
    match_wsi_name_to_transformation_matrix,
    transformation_selection_in_cache)

def test_parse_wsi_path(get_current_dir):
    files = wsi_from_local_path(os.path.join(get_current_dir, 'wsi'))
    assert len(files) == 2
    single = wsi_from_local_path(os.path.join(get_current_dir, 'wsi', 'example_2.svs'))
    assert len(single) == 1
    assert not wsi_from_local_path(os.path.join(get_current_dir, 'wsi', 'example_3.txt'))
    assert not wsi_from_local_path(os.path.join(get_current_dir, 'steinbock'))

def test_update_register_hash(get_current_dir):
    new_hash = update_wsi_hash({}, os.path.join(get_current_dir, 'for_quant.tiff'))
    assert 'for_quant' in new_hash.keys()
    new_hash_2 = update_wsi_hash({}, os.path.join('not_dir', 'for_recolour.tiff'))
    assert isinstance(new_hash_2, dash._callback.NoUpdate)
    new_hash = update_wsi_hash(new_hash, os.path.join(get_current_dir, 'for_recolour.tiff'))
    assert len(new_hash) == 2
    assert 'for_recolour' in new_hash.keys()

    new_hash_list = update_wsi_hash({}, [os.path.join(get_current_dir, 'for_quant.tiff'),
                                         os.path.join(get_current_dir, 'for_recolour.tiff')])
    assert len(new_hash_list) == 2
    assert 'for_recolour' in new_hash_list.keys()

    assert isinstance(update_wsi_hash({}, None), dash._callback.NoUpdate)

def test_import_wsi_transform(get_current_dir):
    transform_hash = update_wsi_hash({}, os.path.join(get_current_dir, 'affine_transform.csv'), True)
    assert 'affine_transform' in transform_hash
    assert isinstance(transform_hash['affine_transform'], np.ndarray)
    assert transform_hash['affine_transform'].shape == (3, 3)

@pytest.mark.skipif(platform.system() != 'Linux',
                    reason='install pyvips only for Linux during testing')
def test_generate_dzi_tiles(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_dir = os.path.join(tmpdirname, "fdsdfsdlfkdn", 'downloads')
        assert not os.path.isfile(os.path.join(download_dir, 'coregister.dzi'))
        dzi_tiles_from_image_path(os.path.join(get_current_dir, 'for_quant.tiff'),
                                  download_dir)
        assert os.path.isfile(os.path.join(download_dir, 'coregister.dzi'))
        tree = ET.parse(os.path.join(download_dir, 'coregister.dzi'))
        root = tree.getroot()
        namespace = root.tag.split('}')[0].strip('{')
        ns_map = {'dz': namespace}
        size_element = root.find('dz:Size', ns_map)
        width = int(size_element.get('Width'))
        height = int(size_element.get('Height'))
        array = tifffile.imread(os.path.join(get_current_dir, 'for_quant.tiff'))
        assert width == array.shape[1]
        assert height == array.shape[0]
        assert os.path.isdir(os.path.join(download_dir, 'coregister_files'))

        # run a second time with the same naming, but different image
        dzi_tiles_from_image_path(os.path.join(get_current_dir, 'for_recolour.tiff'),
                                  download_dir)
        tree = ET.parse(os.path.join(download_dir, 'coregister.dzi'))
        root = tree.getroot()
        namespace = root.tag.split('}')[0].strip('{')
        ns_map = {'dz': namespace}
        size_element = root.find('dz:Size', ns_map)
        width = int(size_element.get('Width'))
        height = int(size_element.get('Height'))
        assert width == height == 600
        assert os.path.isfile(os.path.join(download_dir, 'coregister.dzi'))
        if os.access(download_dir, os.W_OK):
            shutil.rmtree(download_dir)
        assert not os.path.isdir(download_dir)

        dzi_tiles_from_image_path(os.path.join(get_current_dir, 'fake.tiff'),
                                  download_dir)

        assert not os.path.isfile(os.path.join(download_dir, 'coregister.dzi'))

def test_transform_name_in_cache():
    assert not transformation_selection_in_cache(None, None)
    assert not transformation_selection_in_cache(None ,"transform_1")
    assert not transformation_selection_in_cache({"transform": np.ones(100)}, None)
    assert transformation_selection_in_cache({"transform": np.ones(100)}, "transform") is not None
    assert not transformation_selection_in_cache({"transform": np.ones(100)}, "transform_1")


def test_name_match_wsi_transform():
    wsi = "Lung_V1_FFPE"
    transform_options = ['Xenium_Skin_FFPE_V1', 'Lung_v2', 'Pancreas_FFPE', 'Lung_V1_FFPEalign']
    assert match_wsi_name_to_transformation_matrix(wsi, transform_options)
    assert not match_wsi_name_to_transformation_matrix(wsi, list(transform_options[-1]))
    assert not match_wsi_name_to_transformation_matrix(wsi, None)
    assert not match_wsi_name_to_transformation_matrix('Lung_V1', ['Lung_v2', 'Lung_FFPEalign'])
