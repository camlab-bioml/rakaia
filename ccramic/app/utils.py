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


def generate_tiff_stack(tiff_dict, tiff_list):
    print(tiff_list)
    height, width = tiff_dict[tiff_list[0]].shape
    tiffarray = np.zeros((height, width, 3))
    for stack in tiff_list:
        tiffarray[:, :, random.randint(0, 1)] = tiff_dict[stack]
        tiffarray = rescale(tiffarray, 1)

    return Image.fromarray(np.array(tiffarray, dtype='uint8'))


