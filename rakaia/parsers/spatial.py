from typing import Union
import anndata as ad
import numpy as np
import skimage

class SpatialDefaults:
    """
    Defines the default spot sizes and scale numerators for the capture areas
    for both the 6.5um and 11um capture slides. The default spot size in microns is the same for both
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

def is_visium_anndata(adata: ad.AnnData):
    """
    Detect if an anndata object is from the 10X Visium spatial gene expression assay
    """
    return ('spatial' in adata.obsm and 'array_col' in adata.obs and
            'array_row' in adata.obs and visium_has_scaling_factors(adata))

def is_spatial_dataset(adata: ad.AnnData):
    """
    Detect if the anndata passed is a spatial dataset
    """
    return 'spatial' in adata.obsm


def detect_spatial_capture_size(adata: ad.AnnData):
    """
    Get the capture area scale factor for a Visium anndata (65 for 6.5um, and 110 for 11um)
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
    return SpatialDefaults.visium_spot_size if is_visium_anndata(adata) else SpatialDefaults.other_spatial_size

def set_spatial_scale(adata: ad.AnnData, spot_size: Union[int, None]=None, downscale: bool=True):
    """
    Set the Visium scale factors and capture areas based on the metadata in the anndata object as well
    as a custom user provided spot size
    """
    capture_size = detect_spatial_capture_size(adata)
    spot_size = get_spatial_spot_radius(adata, spot_size)
    scale_factor = float(capture_size / (spot_size * 2)) if \
        (downscale and is_visium_anndata(adata)) else 1
    return capture_size, spot_size, scale_factor

def spatial_canvas_dimensions(adata: Union[ad.AnnData, str], border_percentage: float=0.05,
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

        # give a 5% offset around the spots
        y_min = int(y_min - int(border_percentage * (y_max - y_min)))
        x_min = int(x_min - int(border_percentage * (x_max - x_min)))

        y_max = int(y_max + int(border_percentage * (y_max - y_min)))
        x_max = int(x_max + int(border_percentage * (x_max - x_min)))

        # Create an empty grid to hold the gene expression values
        grid_width = int(x_max - x_min)
        grid_height = int(y_max - y_min)
        return grid_width, grid_height, x_min, y_min
    return None, None, None, None

def spatial_grid_single_marker(adata: Union[ad.AnnData, str], gene_marker: str,
                               spot_size: Union[int, None]=None, downscale: bool=True):
    """
    Extracts spot values for a specific gene marker and arranges them in a 2D grid
    based on the spatial coordinates.

    Parameters:
    adata (AnnData): The AnnData object containing spatial data.
    gene_marker (str): The name of the gene marker to extract.

    Returns:
    np.ndarray: A 2D array where each (x, y) position corresponds to the gene expression value.
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    # Ensure the gene exists in the dataset
    if gene_marker not in adata.var_names:
        raise ValueError(f"Gene marker '{gene_marker}' not found in the dataset.")

    # Extract the expression data for the specified gene marker
    gene_index = adata.var_names.get_loc(gene_marker)
    spot_values = adata[:, gene_index].X

    # Convert to dense array if the data is sparse
    if hasattr(spot_values, "toarray"):
        spot_values = spot_values.toarray().flatten()
    else:
        spot_values = spot_values.flatten()

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
    Check the current raw image dictionary for missing visium spot arrays using hte currently selected
    marker list. If markers are missing, add the spot grids to the dictionary
    """
    for selection in channel_list:
        if image_dict[data_selection][selection] is None:
            image_dict[data_selection][selection] = spatial_grid_single_marker(adata, selection,
                                                                               spot_size)
    return image_dict


def get_spatial_spot_radius(adata: ad.AnnData, alt_spot_size: Union[int, None]=None):
    """
    Parse the Visium Anndata object for the spot size relative to the original image,
    used to render the spots in pixel format. If the value cannot be found, then the default is 55
    """
    # if a custom spot size if provided, use that first
    if alt_spot_size:
        return alt_spot_size
    try:
        spatial_meta = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]
        return int(spatial_meta['scalefactors']['spot_diameter_fullres'] / 2)
    except KeyError:
        return alt_spot_size if alt_spot_size else set_spatial_spot_size(adata)
