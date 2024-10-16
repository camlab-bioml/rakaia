import anndata as ad
import numpy as np
import skimage

def visium_canvas_dimensions(adata, border_percentage: float=0.05):
    """
    Set the dimensions for the 10x visium data set. Incorporates a border percentage that
    is the proportion of the 10x visium resolution that should form a border around the spots
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    if adata and 'spatial' in adata.obsm:
        x_min, y_min = np.min(adata.obsm['spatial'], axis=0)
        x_max, y_max = np.max(adata.obsm['spatial'], axis=0)

        # give a 5% offset around the spots
        y_min = int(y_min - int(border_percentage * (y_max - y_min)))
        x_min = int(x_min - int(border_percentage * (x_max - x_min)))

        y_max = int(y_max + int(border_percentage * (y_max - y_min)))
        x_max = int(x_max + int(border_percentage * (x_max - x_min)))

        # Create an empty grid to hold the gene expression values
        grid_width = int(x_max - x_min)
        grid_height = int(y_max - y_min)
        return grid_width, grid_height, x_min, y_min
    return None

def visium_spot_grid_single_marker(adata, gene_marker, spot_size: int=55):
    """
    Extracts spot values for a specific gene marker and arranges them in a 2D grid
    based on the spatial coordinates.

    Parameters:
    adata (AnnData): The AnnData object containing 10x Visium spatial data.
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

    spatial_coords = adata.obsm['spatial']
    # spatial_coords = np.column_stack((adata.obs['array_col'], adata.obs['array_row']))

    grid_width, grid_height, x_min, y_min = visium_canvas_dimensions(adata)
    gene_grid = np.zeros((grid_height, grid_width))

    # Map the gene expression values to the grid
    for i, (x, y) in enumerate(spatial_coords):
        grid_x = int(x - x_min)  # Shift the x-coordinate to start from 0
        grid_y = int(y - y_min)  # Shift the y-coordinate to start from 0
        try:
            gene_grid[skimage.draw.disk((grid_y, grid_x), radius=int(spot_size))] = float(spot_values[i])
        except IndexError:
            pass

    return gene_grid.astype(np.float32)


def check_spot_grid_multi_channel(image_dict, data_selection, adata, channel_list):
    """
    Check the current raw image dictionary for missing visium spot arrays
    """
    for selection in channel_list:
        if image_dict[data_selection][selection] is None:
            image_dict[data_selection][selection] = visium_spot_grid_single_marker(adata, selection)
    return image_dict
