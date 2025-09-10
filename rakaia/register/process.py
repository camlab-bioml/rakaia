"""Module defining the functions for processing images for registration such as H & E
"""
import os
from typing import Union
import shutil
from pathlib import Path
import dash
from rakaia.io.session import create_download_dir
from rakaia.utils.pixel import high_low_values_from_zoom_layout

WSI_FILE_EXTENSIONS = ['tif', 'tiff', 'svs', 'btf', 'ndpi', 'scn']

def wsi_from_local_path(path: str):
    """
    Parse a local filepath, either filename or directory, for WSI compatible files
    """
    if os.path.isfile(path) and any([path.endswith(ext) for ext in WSI_FILE_EXTENSIONS]):
        return [path]
    elif os.path.isdir(path):
        return [str(os.path.join(path, file)) for
                file in os.listdir(path) if any([file.endswith(ext)
                for ext in WSI_FILE_EXTENSIONS])]
    return None

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

def coordinate_scaling(bounds: dict, scaling_val: float=0.2125,
                       dim_normalize: Union[list, tuple, None]=None):
    """
    Define osd-compatible coordinates changed by a scaling factor
    Ir dim normalize is passed (format: [y, x]), normalize by the dimensions to get between 0-1
    """
    x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(bounds)
    height = int(int(y_high * scaling_val) - int(y_low * scaling_val))
    width = int(int(x_high * scaling_val) - int(x_low * scaling_val))
    # dimension normalization must use the scaling value as well as the dimensions to get between 0-1
    if dim_normalize:
        height = height / (int(dim_normalize[0]) * scaling_val)
        width = width / (int(dim_normalize[1]) * scaling_val)
        scaling_return = (f"{x_low / dim_normalize[1]},"
                f"{y_low / dim_normalize[0]},{width},{height}")
        print(scaling_return)
        return scaling_return
    print(f"{x_low * scaling_val},{y_low * scaling_val},{width},{height}")
    return f"{x_low * scaling_val},{y_low * scaling_val},{width},{height}"
