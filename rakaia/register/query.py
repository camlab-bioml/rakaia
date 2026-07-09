"""
Module related to functions and classes for processing WSI patches and enabling queries
"""
import io
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image

def wsi_crop(image: Union[Path, str, np.ndarray],
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
        if crop.shape[2] == 4: crop = crop[:, :, :3]
        # TODO: how should the aspect ratio be handled? Here we make a square subsample patch
        return np.array(Image.fromarray(crop).resize((patch_out_size, patch_out_size),
             resample=Image.Resampling.LANCZOS)) if (return_sampled and patch_out_size) else crop
    except pyvips.Error: pass
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
