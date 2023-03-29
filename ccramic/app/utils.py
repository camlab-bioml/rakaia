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
from PIL import ImageColor


def get_luma(rbg):
    R, G, B, = rbg
    return 0.2126*R + 0.7152*G + 0.0722*B


def generate_tiff_stack(tiff_dict, tiff_list, colour_dict):
    # image = recolour_greyscale(tiff_dict[tiff_list[0]], colour_dict[tiff_list[0]])
    # for other in tiff_list[1:]:
    #     image = image + recolour_greyscale(tiff_dict[other], colour_dict[other]
    return Image.fromarray(sum([recolour_greyscale(tiff_dict[elem], colour_dict[elem]) for elem in tiff_list]))


def recolour_greyscale(array, colour):
    image = Image.fromarray(array)
    image = image.convert('RGB')
    pixels = image.load()
    r, g, b = ImageColor.getcolor(colour, "RGB")
    for i in range(image.height):
        for j in range(image.width):
            try:
                luma = get_luma(pixels[i, j])
                pixels[i, j] = (int(r * (luma / 255)), int(g * (luma / 255)), int(b * (luma / 255)))
            except ZeroDivisionError:
                pixels[i, j] = (0, 0, 0)

    return np.array(image)

