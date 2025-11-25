"""Module defining the functions for processing images for registration such as H & E
"""
import os
from typing import Union
import shutil
from pathlib import Path
import dash
import pandas as pd
import numpy as np

from rakaia.io.session import create_download_dir

WSI_FILE_EXTENSIONS = ['tif', 'tiff', 'svs', 'btf', 'ndpi', 'scn', 'TIF', 'TIFF']

def wsi_from_local_path(path: str):
    """
    Parse a local filepath, either filename or directory, for WSI compatible files
    """
    if os.path.isfile(path) and any([path.endswith(ext) for ext in WSI_FILE_EXTENSIONS]):
        return [path]
    if os.path.isdir(path):
        return list(set([str(os.path.join(path, file)) for
                file in os.listdir(path) if any(file.endswith(ext)
                for ext in WSI_FILE_EXTENSIONS)]))
    return None

def update_wsi_hash(cur_hash: Union[dict, None],
                    new_upload: Union[str, Path, list, None]=None,
                    read_as_array: bool=False):
    """
    Update a dictionary for WSI uploads (either WSI images or transformation matrices in CSV format).
    """
    uploads = [new_upload] if isinstance(new_upload, str) else new_upload
    cur_hash = cur_hash if cur_hash is not None else {}
    if uploads:
        for upload in uploads:
            if upload and os.path.isfile(upload):
                # ignore any extra extensions such as .ome.tiff
                wsi_identifier = str(Path(upload).stem).split(".")[0]
                cur_hash[wsi_identifier] = upload if not read_as_array else (
                    np.array(pd.read_csv(upload, header=None)))
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
    try:
        image = pyvips.Image.new_from_file(image_path, access="sequential")
        try:
            if os.path.exists(os.path.join(dest_dir, f"{static_folder_prefix}_files")):
                shutil.rmtree(os.path.join(dest_dir, f"{static_folder_prefix}_files"))
        except FileNotFoundError: pass
        create_download_dir(os.path.join(dest_dir, static_folder_prefix))
        image.dzsave(os.path.join(os.path.join(dest_dir, static_folder_prefix)),
                     suffix=".jpg", tile_size=256, overlap=1)
    except pyvips.Error: pass

def match_wsi_name_to_transformation_matrix(wsi_name: str, transform_options: Union[list, None]=None):
    """
    Search for a name match among the WSI transformation dropdown options for the currently selected WSI.
    Assumes that the WSI file name is in the imported transformation name, or partial overlap (i.e.
    the entire WSI base name is in the transformation name).
    """
    if wsi_name and transform_options:
        for transform in transform_options:
            if str(wsi_name) in str(transform):
                return str(transform)
        return None
    return None
