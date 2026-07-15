"""
Module related to functions and classes for processing WSI patches and enabling API queries
"""
import io
import copy
from typing import Union
from pathlib import Path
import requests
import numpy as np
from PIL import Image

# define the default column definitions for the TCGA UNI search results shown in dash ag grid
TCGA_UNI_COL_DEFS = [{"field": "project", "rowGroup": True, "hide": True}, {"field": "slide", "rowGroup": True, "hide": True},
                    {"field": "x"}, {"field": "y"},
                    {"field": "url", "cellRenderer": "LinkRenderer"}, {"field": "similarity"}]

def wsi_crop(image: Union[Path, str, np.ndarray, None],
             bounds: Union[list, None]=None,
             return_sampled: bool=True,
             patch_out_size: int=224):
    """
    Generate a crop of a WSI image processed through pyvips. Assumes tha the bounds array is in the
    format `[x0, x1, y0, y1]`.
    If `return_subsample` is used, specify the size (i.e. 224 works for UNI patch embeddings)
    """
    import pyvips
    try:
        x0, x1, y0, y1 = bounds
        crop = pyvips.Image.new_from_file(image, access="sequential") if not \
            isinstance(image, np.ndarray) else pyvips.Image.new_from_array(image, interpretation='rgb')
        crop = crop.crop(x0, y0, x1 - x0, y1 - y0).numpy().astype(np.uint8)
        # drop alpha channel if present, often from svs
        if (len(crop.shape) == 3) and crop.shape[2] == 4: crop = crop[:, :, :3]
        # TODO: how should the aspect ratio be handled? Here we make a square subsample patch
        return np.array(Image.fromarray(crop).resize((patch_out_size, patch_out_size),
             resample=Image.Resampling.LANCZOS)) if (return_sampled and patch_out_size) else crop
    except (pyvips.Error, TypeError, KeyError): pass
    return None

def serialize_crop(crop: Union[np.array, np.ndarray, None]=None):
    """
    Serialize the WSI crop into compressed bytes for a POST request
    """
    if crop is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=np.stack(crop))
        return buffer.getvalue()
    return None

def tcga_uni_request(crop: Union[np.ndarray, np.array, None]=None,
                            api_host: str="localhost",
                            api_port: int=6000,
                            k_search: int=10,
                            return_url: bool=True,
                            endpoint: str="search"):
    """
    Format the TCGA UNI POST request to send to hist2query
    """
    if crop is not None:
        response = requests.post(f"http://{api_host}:{api_port}/{endpoint}",
                                 files={"patch": ("patch.npy", serialize_crop(crop.astype(np.uint8)))},
                                 data={"k": k_search, "url": return_url}, timeout=300)
        response.raise_for_status()
        return response.json()
    return None

def format_col_ag_groupings(use_grouping: bool=True):
    """
    format the col groupings
    """
    new_col_defs = copy.deepcopy(TCGA_UNI_COL_DEFS)
    for col in new_col_defs:
        if "rowGroup" in col:
            col["rowGroup"] = use_grouping
            col["hide"] = use_grouping
    return new_col_defs
