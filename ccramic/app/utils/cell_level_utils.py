import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tifffile
import yaml
from PIL import Image
from PIL import Image
import numpy as np
from pathlib import Path
from tifffile import TiffFile
from ccramic.app.utils.pixel_level_utils import *
import plotly.graph_objects as go
import cv2

def get_pixel(mask, i, j):
    if len(mask.shape) > 2:
        return mask[i][j][0]
    else:
        return mask[i][j]

def convert_mask_to_cell_boundary(mask, outline_color=255, greyscale=True):
    """
    Convert a mask array with filled in cell masks to an array with drawn boundaries with black interiors of cells
    """
    if greyscale:
        outlines = np.full((mask.shape[0], mask.shape[1]), 3)
    else:
        outlines = np.stack([np.empty(mask[0].shape), mask[0], mask[1]], axis=2)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            pixel = get_pixel(mask, i, j)
            if pixel != 0:
                if i != 0 and get_pixel(mask, i - 1, j) != pixel:
                    outlines[i][j] = outline_color
                elif i != mask.shape[0] - 1 and get_pixel(mask, i + 1, j) != pixel:
                    outlines[i][j] = outline_color
                elif j != 0 and get_pixel(mask, i, j - 1) != pixel:
                    outlines[i][j] = outline_color
                elif j != mask.shape[1] - 1 and get_pixel(mask, i, j + 1) != pixel:
                    outlines[i][j] = outline_color

    # Floating point errors can occaisionally put us very slightly below 0
    return np.where(outlines >= 0, outlines, 0).astype(np.uint8)
