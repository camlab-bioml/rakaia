"""Module containing functions and classes for parsing and validating raw spatial dataset
file imports and user-defined spatial visualization preference such as marker size
"""

from typing import Union
import anndata as ad
import numpy as np
import skimage

class SpatialDefaults:
    """
    Defines the default spot/marker sizes and scale numerators for spatial datasets.
    For 10X Visium, the default spot size in microns is the same for both the 6.5um and 11um capture slides.
    For non-Visium technologies, the default marker radius in pixels is 1, and can be changed in-session
    """
    visium_spot_size: int=55
    visium_scale_num: int=65
    other_spatial_size: int=1

def visium_has_scaling_factors(adata: ad.AnnData):
    """
    Return if the Anndata parsed has scaling factor information present in the
    """
    try:
        spatial_meta = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]
        return 'scalefactors' in spatial_meta.keys()
    except KeyError:
        return False

def is_spot_based_spatial(adata: Union[ad.AnnData, str]):
    """
    Detect if an anndata object is from the 10X Visium spatial gene expression spot-based assay.
    This includes Visium V1 & V2, but not Visium HD, which uses square bins
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    return ('spatial' in adata.obsm and 'array_col' in adata.obs and
            'array_row' in adata.obs and visium_has_scaling_factors(adata))

def is_spatial_dataset(adata: ad.AnnData):
    """
    Detect if the anndata passed is a spatial dataset
    """
    return 'spatial' in adata.obsm


def detect_spatial_capture_size():
    """
    Get the capture area scale factor for a Visium spot-based dataset. This is currently set to 65
    for both capture area resolutions (6.5 and 11mm)
    """
    # try:
    #     return VisiumDefaults.scale_num if (int(np.max(adata.obs['array_col'])) > 200 and
    #             int(np.max(adata.obs['array_row'] > 100))) else VisiumDefaults.scale_num
    # except KeyError:
    #     return VisiumDefaults.scale_num
    # appears that both 6.5 and 11 um capture areas use the same normalization factor
    # https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
    return SpatialDefaults.visium_scale_num

def set_spatial_spot_size(adata):
    """
    Set the marker spot size for a spatial dataset
    """
    return SpatialDefaults.visium_spot_size if is_spot_based_spatial(adata) else SpatialDefaults.other_spatial_size

def set_spatial_scale(adata: ad.AnnData, spot_size: Union[int, None]=None, downscale: bool=True):
    """
    Set the Visium scale factors and capture areas based on the metadata in the anndata object as well
    as a custom user provided spot size
    """
    capture_size = detect_spatial_capture_size()
    spot_size = get_spatial_spot_radius(adata, spot_size)
    scale_factor = float(capture_size / (spot_size * 2)) if \
        (downscale and is_spot_based_spatial(adata)) else 1
    return capture_size, spot_size, scale_factor

def spatial_canvas_dimensions(adata: Union[ad.AnnData, str], border_percentage: float=0.03,
                              downscale: bool=True):
    """
    Set the dimensions for the 10x visium data set. Incorporates a border percentage that
    is the proportion of the 10x visium resolution that should form a border around the spots
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    if adata and 'spatial' in adata.obsm:
        capture_size, spot_size, scale_factor = set_spatial_scale(adata, downscale=downscale)

        adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
        x_min, y_min = np.min((adata.obsm['spatial'] * scale_factor), axis=0)
        x_max, y_max = np.max((adata.obsm['spatial'] * scale_factor), axis=0)

        # only use a border for visium because of the masks being able to match for other technologies
        if is_spot_based_spatial(adata):
            y_min = int(y_min - int(border_percentage * (y_max - y_min)))
            x_min = int(x_min - int(border_percentage * (x_max - x_min)))

            y_max = int(y_max + int(border_percentage * (y_max - y_min)))
            x_max = int(x_max + int(border_percentage * (x_max - x_min)))

        # Create an empty grid to hold the gene expression values
        grid_width = int(x_max - x_min)
        grid_height = int(y_max - y_min)
        return grid_width, grid_height, x_min, y_min
    return None, None, None, None

def spatial_marker_to_dense_flat(spot_array: Union[np.array, np.ndarray]):
    """
    Convert a sparse spatial marker array into a dense flat array
    """
    if hasattr(spot_array, "toarray"):
        spot_array = spot_array.toarray().flatten()
    else:
        spot_array = spot_array.flatten()
    return spot_array

def spatial_grid_single_marker(adata: Union[ad.AnnData, str], gene_marker: Union[str, None],
                               spot_size: Union[int, None]=None, downscale: bool=True,
                               as_mask: int=False):
    """
    Extracts spot values for a specific gene marker and arranges them in a 2D grid
    based on the spatial coordinates. Requires either a named marker for expression spots,
    or the `as_mask` flag to render spot detection for all spots. The `downscale` flag controls if
    scale factors are used to match the image to the experiment capture area and should be used.
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    # Ensure the gene exists in the dataset
    if gene_marker not in adata.var_names and not as_mask:
        raise ValueError(f"Gene marker '{gene_marker}' not found in the dataset.")

    # Extract the expression data for the specified gene marker
    spot_values = np.array(range(1, (len(adata) + 1))) if as_mask else (
        adata[:, adata.var_names.get_loc(gene_marker)].X)

    # Convert to dense array if the data is sparse
    spot_values = spatial_marker_to_dense_flat(spot_values)

    # Extract spatial coordinates from the 'spatial' key in .obsm
    if 'spatial' not in adata.obsm.keys():
        raise ValueError("Spatial coordinates not found in 'obsm' attribute.")

    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
    capture_size, spot_size, scale_factor = set_spatial_scale(adata, spot_size, downscale=downscale)
    spot_size = spot_size if not downscale else int(spot_size * scale_factor)

    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(adata, downscale=downscale)
    gene_grid = np.zeros((grid_height, grid_width))

    spatial_coords = np.array(adata.obsm['spatial'] * scale_factor).astype(np.float32)
    # Map the gene expression values to the grid
    for i, (x, y) in enumerate(spatial_coords):
        if float(spot_values[i]) > 0:
            grid_x = int(x - x_min)  # Shift the x-coordinate to start from 0
            grid_y = int(y - y_min)  # Shift the y-coordinate to start from 0
            try:
                gene_grid[skimage.draw.disk((grid_y, grid_x), radius=int(spot_size))] = float(spot_values[i])
            except IndexError: pass

    return gene_grid.astype(np.float32)


def check_spot_grid_multi_channel(image_dict: dict, data_selection: str,
                                  adata: ad.AnnData, channel_list: list,
                                  spot_size: Union[int,float]=55):
    """
    Check the current raw image dictionary for missing visium spot arrays using the currently selected
    marker list. If markers are missing, add the spot grids to the dictionary
    """
    for selection in channel_list:
        if not image_dict[data_selection] or image_dict[data_selection][selection] is None:
            image_dict[data_selection][selection] = spatial_grid_single_marker(adata, selection, spot_size)
    return image_dict


def get_spatial_spot_radius(adata: ad.AnnData, alt_spot_size: Union[int, None]=None):
    """
    Parse the Visium Anndata object for the spot size relative to the original image,
    used to render the spots in pixel format. If the value cannot be found, then the default is 55
    """
    # if a custom spot size if provided, use that first
    if alt_spot_size and not is_spot_based_spatial(adata):
        return alt_spot_size
    try:
        spatial_meta = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]
        return int(spatial_meta['scalefactors']['spot_diameter_fullres'] / 2)
    except KeyError:
        return alt_spot_size if (alt_spot_size and not is_spot_based_spatial(adata)) else (
            set_spatial_spot_size(adata))
