"""Module defining the functions for processing images for registration such as H & E
"""
import os
from typing import Union
import shutil
from pathlib import Path
import dash
from rakaia.io.session import create_download_dir

def update_coregister_hash(cur_hash: Union[dict, None],
                           new_upload: Union[str, Path, list, None]=None):
    """
    Update the co-register hash with new file upload
    """
    uploads = [new_upload] if isinstance(new_upload, str) else new_upload
    cur_hash = cur_hash if cur_hash is not None else {}
    if uploads:
        for upload in uploads:
            if upload and os.path.isfile(upload):
                cur_hash[os.path.basename(upload)] = upload
        return cur_hash if cur_hash else dash.no_update
    return dash.no_update

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
        if os.path.exists(os.path.join(dest_dir, f"{static_folder_prefix}_files")):
            shutil.rmtree(os.path.join(dest_dir, f"{static_folder_prefix}_files"))
    except FileNotFoundError:
        pass
    create_download_dir(os.path.join(dest_dir, static_folder_prefix))
    image.dzsave(os.path.join(os.path.join(dest_dir, static_folder_prefix)),
                 suffix=".jpg", tile_size=256, overlap=1)
