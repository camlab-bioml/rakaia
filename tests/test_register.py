import tempfile
import xml.etree.ElementTree as ET
import os
import platform
import pytest
import shutil
import tifffile
from rakaia.register.dzi import dzi_tiles_from_image_path

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
        if os.access(download_dir, os.W_OK):
            shutil.rmtree(download_dir)
        assert not os.path.isdir(download_dir)
