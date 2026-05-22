"""
Module defining functions and classes for transferring coordinates between the multiplexed
canvas and the WSI OSD view.
"""
from typing import Union
import anndata as ad
import numpy as np
from numpy.linalg import inv
import pandas as pd
from rakaia.parsers.spatial import spatial_canvas_dimensions, trim_neg_val
from rakaia.utils.pixel import high_low_values_from_zoom_layout

class WSICanvasAffineCoordTransfer:
    """
    Transfer a set of coordinates from the blend canvas to the WSI viewer using an affine transformation matrix.

    :param bounds: Dictionary of x and y-axis min and max bounds for the current canvas zoom level
    :param spatial_anndata: If an Anndata/.h5ad dataset is used, provide the pointer to compute the actual x and y min
    :param transformation_matrix: numpy array or path to a `pandas` CSV containing an affine transformation
    """
    def __init__(self, bounds: dict,
                spatial_anndata: Union[ad.AnnData, str, None],
                transformation_matrix: Union[np.ndarray, str]):
        self.bounds = bounds
        self.adata = spatial_anndata if (isinstance(spatial_anndata, ad.AnnData) or
                    (isinstance(spatial_anndata, str) and str(spatial_anndata).endswith('.h5ad'))) else None
        self.transform = np.array(pd.read_csv(transformation_matrix, header=None)) if not (
            isinstance(transformation_matrix, np.ndarray)) else transformation_matrix

    def process_coordinates(self, scaling_factor: float=0.21,
                            use_inverse: Union[bool, str, None]=True):
        """
        Process the coordinates given from the canvas with a designated scaling factor for a matched WSI view

        :param scaling_factor: Pixel scaling factor denoting the microns per pixel conversion for the WSI.
        :param use_inverse: Whether to use the inverse transform or not. Use the inverse if the matrix maps WSI -> canvas.

        :return: String [x_min,y_min,width,height] in pixels of the matched bound in the WSI view, compatible with osd JS.
        """
        # https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
        # need to compute the actual tissue x and y min from the Anndata coordinates
        grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(self.adata) if (
            self.adata) else (0, 0, 0, 0)
        x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(self.bounds)

        matrix_use = inv(self.transform) if use_inverse else self.transform
        # get the bounds for two opposite corners to get the full bound
        both_low = np.matmul(matrix_use,
                             np.array([trim_neg_val((x_low + x_min) / scaling_factor),
                                       trim_neg_val((y_low + y_min) / scaling_factor), 1]))

        both_high = np.matmul(matrix_use,
                              np.array([trim_neg_val((x_high + x_min) / scaling_factor),
                                        trim_neg_val((y_high + y_min) / scaling_factor), 1]))

        concat = np.delete(np.vstack((both_low, both_high)), -1, axis=1)
        out_x_min, out_y_min = np.min(concat, axis=0)
        out_x_max, out_y_max = np.max(concat, axis=0)
        height = int(out_y_max - out_y_min)
        width = int(out_x_max - out_x_min)
        return f"{out_x_min},{out_y_min},{width},{height}"
