import tifffile
from tifffile import TiffFile
from imctools.converters import ome2analysis
from imctools.converters import ome2histocat
from imctools.converters import mcdfolder2imcfolder
from imctools.converters import exportacquisitioncsv
import os
from os import listdir
import pathlib
import shutil
import re
import pandas as pd
import numpy as np
import skimage
from skimage.transform import rescale, resize
from skimage import exposure
from numpy import array
from glob import glob
from shutil import copyfile
from skimage import exposure
from numpy import array
from skimage.io import imread, imsave
from PIL import Image
import random

def get_luma(rbg):
    R, G, B, = rbg
    return 0.2126*R + 0.7152*G + 0.0722*B


def generate_tiff_stack(tiff_dict, tiff_list):
    image = tiff_dict[tiff_list[0]]
    for other in tiff_list[1:]:
        image = image + tiff_dict[other]

    return Image.fromarray(image).convert('RGB')

def recolour_greyscale(image, colour):
    pixels = image.load()
    print(pixels)
    for i in range(image.height):
        for j in range(image.width):
            luma = get_luma(pixels[i, j])
            if luma > 0.05:
                pixels[i, j] = colour
            else:
                pixels[i, j] = (0, 0, 0)

    return np.array(image)

