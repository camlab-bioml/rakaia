"""
Module for classes and functions related to creating stitched Whole slide Images (WSI)
from canvas blends
"""

from typing import Union
import shutil
import os
import numpy as np
import tifffile
from PIL import Image
import dash_bootstrap_components as dbc
from rakaia.components.canvas import CanvasImage
from rakaia.utils.object import check_download_dir
from rakaia.utils.pixel import resize_for_canvas

def update_stitch_cache_with_blend(stitch_cache: Union[dict, None],
                                   stitch_selection: Union[str, None],
                                   stitch_x_coord: Union[int, float, None]=None,
                                   stitch_y_coord: Union[int, float, None]=None,
                                   roi_to_add: Union[np.ndarray, CanvasImage, None]=None):
    """
    Update a selected stitched image in the cache with an ROI blend at a specific x and y-min.
    This adds the RGB blend in that position and returns the updated cache, then cached
    as a server side transformed data object
    """
    if None not in (stitch_cache, stitch_selection, stitch_x_coord, stitch_y_coord) and str(stitch_selection) in stitch_cache:
        try:
            roi_to_add = roi_to_add.get_image() if not isinstance(roi_to_add, np.ndarray) else roi_to_add
            stitch_cache[stitch_selection][stitch_y_coord:(roi_to_add.shape[0] + int(stitch_y_coord)),
            stitch_x_coord:(roi_to_add.shape[1] + int(stitch_x_coord))] = roi_to_add
        except ValueError: pass
    return stitch_cache

def stitch_cache_dropdown_labels(stitch_cache: Union[dict, None]):
    """
    Generate the stitch cache dropdown labels to include the dimensions of the created images
    in the label
    """
    options = []
    if stitch_cache is not None and isinstance(stitch_cache, dict):
        for key, value in stitch_cache.items():
            if isinstance(value, np.ndarray):
                options.append({'label': f'{str(key)} ({value.shape[1]}x{value.shape[0]})',
                                'value': str(key)})
    return options

def download_stitch_image(dest_dir: Union[str, None]=None,
                          stitch_cache: Union[dict, None]=None,
                          stitch_selection: Union[str, None]=None):
    """
    Create a zipped archive for a specific stitched image and return the path, compatible with `dcc.send_file`
    """
    if None not in (dest_dir, stitch_cache, stitch_selection) and stitch_selection in stitch_cache:
        check_download_dir(dest_dir, True)
        tifffile.imwrite(os.path.join(dest_dir, f"{stitch_selection}.tiff"),
                         stitch_cache[stitch_selection].astype(np.uint8))
        shutil.make_archive(dest_dir, 'zip', dest_dir)
        return str(dest_dir + ".zip")


def stitch_image_preview(stitch_collection: Union[dict, None]=None,
                          stitch_selection: Union[str, None]=None):
    """
    Generate a thumbnail preview for a selected stitch image, displayed as a down-sampled Card. Return
    a tuple toggling the visibility of the parent modal, and the card HTML
    """
    if None not in (stitch_collection, stitch_selection) and str(stitch_selection) in list(stitch_collection.keys()):
        return True, [dbc.Col(dbc.Card([dbc.CardImg(src=Image.fromarray(resize_for_canvas(
        stitch_collection[stitch_selection], 1000)).convert('RGB'), bottom=True)]), width=12)]
    return False, None
