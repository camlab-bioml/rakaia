import os
from typing import Union
import shutil
from pathlib import Path
from rakaia.io.session import create_download_dir

def dzi_tiles_from_image_path(image_path: Union[Path, str],
                              dest_dir: Union[Path, str],
                              static_folder_prefix: str="coregister"):
    """
    Use `pyvips` to generate a series of dzi tiles that can be served to the flask static route
    Use the `static_folder_prefix` to match the dzi and tiles to `openseadragon`
    """
    import pyvips
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    try:
        shutil.rmtree(os.path.join(dest_dir))
    except FileNotFoundError:
        pass
    create_download_dir(os.path.join(dest_dir, static_folder_prefix))
    image.dzsave(os.path.join(os.path.join(dest_dir, static_folder_prefix)),
                 suffix=".jpg", tile_size=256, overlap=1)
