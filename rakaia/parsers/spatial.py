"""Module containing functions and classes for parsing and validating raw spatial dataset
file imports and user-defined spatial visualization preference such as marker size
"""
import os
from typing import Union
import anndata as ad
import numpy as np
import rasterio
from numpy.linalg import inv
import pandas as pd
import skimage
from pathlib import Path
import spatialdata as sd
from rasterio.features import rasterize
from tifffile import imwrite
import dash
from shapely import affinity
from rakaia.utils.pixel import split_string_at_pattern, high_low_values_from_zoom_layout, \
    set_array_storage_type_from_config
from rakaia.utils.session import validate_session_upload_config

class ZarrSDKeys:
    """
    Define the identifying subdirectories for spatialdata stores as well as technology-specific
    terms to recognize 10X Genomics assays
    """
    dirs_include = ['images', 'points', 'shapes', 'tables']
    visium_hd_tables = ['square_002um', 'square_008um', 'square_016um']
    xenium_shapes = ['cell_boundaries', 'cell_circles', 'nucleus_boundaries']
    visium_hd_bin_sizes = ["8", "16"]


def is_zarr_store(local_dir: Union[Path, str]):
    """
    Define if a provided directory is a zarr store
    """
    # Case 1: if the base directory has either of these files, say it is a zarr store
    if (os.path.exists(os.path.join(local_dir, 'zmetadata')) or
            os.path.exists(os.path.join(local_dir, '.zgroup'))):
        return True
    # Case 2: search in any of the potential spatialdata sub-directories for the .zgroup file
    if any(os.path.exists(os.path.join(local_dir, str_dir, '.zgroup')) for
           str_dir in ZarrSDKeys.dirs_include):
        return True
    return False

class ZarrSDParser:
    """
    Parse a spatialdata zarr-backed store for 10X Genomics ST outputs. Currently, supports parsing
    for 10X Visium, Visium HD, and Xenium

    :param zarr_path: local directory path to the zarr store
    :param tmp_session_path: In-application path to where temporary spatial files should be written
    :param cur_session_uploads: Dictionary of current imaging uploads in the session if they exist, or `None`
    :param cur_mask_uploads: Dictionary of current mask uploads in the session if they exist, or `None`

    :return: None
    """
    def __init__(self, zarr_path: Union[Path, str, None]=None,
                 tmp_session_path: Union[Path, str, None]=None,
                 cur_session_uploads: Union[dict, None]=None,
                 cur_mask_uploads: Union[dict, None]=None):

        self._zarr_path = str(Path(zarr_path).resolve())
        self._tmp_session_path = tmp_session_path
        # make the outputs match the `parse_steinbock_dir` output format/order
        self._image_paths = validate_session_upload_config(cur_session_uploads)
        self._mask_paths = {} if (cur_mask_uploads is None or
                                not isinstance(cur_mask_uploads, dict)) else cur_mask_uploads
        self._quant = None
        self._error = None
        self._umap = None
        self._scaling = None

    @staticmethod
    def truthy_no_update(val: Union[dict, None]=None):
        """
        Return a truthy value, otherwise return a `no_update` object

        :param val: Dictionary value to return

        :return: The dictionary value or a `dash.no_update` if not truthy
        """
        return val if val is not None else dash.no_update

    def check_session_cache(self):
        """
        Check if the provided session path exists. If not, create it

        :return: None
        """
        if not os.path.exists(self._tmp_session_path):
            os.makedirs(self._tmp_session_path)

    def write_adata(self, adata: ad.AnnData, sample_id: str):
        """
        Write an anndata to the in-session temporary storage

        :param adata: `Anndata` containing spatial expression profiles
        :param sample_id: Identifier to tag the outgoing `.h5ad`

        :return: Output path for the expression `.h5ad` written to the session tmp directory
        """
        self.check_session_cache()
        out_path = f"{self._tmp_session_path}/{sample_id}.h5ad"
        adata.write_h5ad(Path(out_path))
        return out_path

    def write_mask(self, mask_array: Union[np.array, np.ndarray],
                   sample_id: str):
        """
        Write a segmentation mask to the in-application temporary storage

        :param mask_array: `numpy` mask array with integers as mask object identifiers in `np.uint32` format
        :param sample_id: Identifier to tag the outgoing mask

        :return: Output path for the mask tiff written to the session tmp directory
        """
        self.check_session_cache()
        out_path = f"{self._tmp_session_path}/{sample_id}.tiff"
        imwrite(out_path, mask_array.astype(np.uint32))
        return out_path

    @staticmethod
    def is_visium_hd(sdset: sd.SpatialData):
        """
        Determine if the spatialdata object has table keys corresponding to 10X Visium HD

        :param sdset: `spatialdata` object containing either ST or multiplexed imaging measurements

        :return: Boolean if any of the 10X Visium HD column identifiers are found in the tables
        """
        return any(col_id in sdset.tables for col_id in ZarrSDKeys.visium_hd_tables)

    @staticmethod
    def scale_visium_hd_by_bin(adata: ad.AnnData, bin_size: int):
        """
        Scale the expression of a Visium HD dataset by the bin size

        :param adata: `Anndata` containing spatial expression profiles
        :param bin_size: Integer specifying the corresponding bin size to scale the spatial coordinates

        :return: Scaled `Anndata` containing spatial coordinates scaled by the bin size
        """
        adata.obsm['spatial'] = adata.obsm['spatial'] / float(bin_size)
        adata.var_names_make_unique()
        adata.uns["scaling_visium_hd"] = int(bin_size)
        return adata.copy()

    @staticmethod
    def is_xenium(sdset: sd.SpatialData):
        """
        Determine if the spatialdata object has shape keys corresponding to 10X Xenium

        :param sdset: `spatialdata` object containing either ST or multiplexed imaging measurements

        :return: Boolean if any of the 10X Xenium file identifiers are found in the shapes
        """
        return any(col_id in sdset.shapes for col_id in ZarrSDKeys.xenium_shapes)

    @staticmethod
    def spatial_segmentation_mask(sdset: sd.SpatialData,
                                  flip: bool=True,
                                  shape_key: str="cell_boundaries",
                                  table_key: str="table",
                                  scale_factor: int | None=None):
        """
        Generate a spatial segmentation mask from a shape frame with a matching expression table

        :param sdset: `spatialdata` object containing either ST or multiplexed imaging measurements
        :param flip: Whether to flip the mask along the y-axis due to rasterization
        :param shape_key: Key for the shape in the sdata `shapes` slot. Provides the segmentation polygons.
        :param table_key: Key for the table in the sdata `tables` slot. Provides the coordinate limits for the output mask.
        :param scale_factor: Integer value for scaling the mask objects. Default is `None`
        :return: Numpy mask array with object (i.e. cell) boundaries and object ids of the type `np.uint32`
        """
        if shape_key in sdset.shapes and table_key in sdset.tables:
            adata = sdset.tables[table_key]
            cells = sdset.shapes[shape_key]

            # get the int bounds to know where the segmentation mask is
            x_min, y_min = np.min((adata.obsm['spatial']), axis=0)
            x_max, y_max = np.max((adata.obsm['spatial']), axis=0)

            transform = rasterio.transform.from_bounds(x_min, y_min, x_max,
                        y_max, int(x_max - x_min), int(y_max - y_min))

            if scale_factor:
                # scale the segmentation by the bin size (i.e. Visium HD)
                cells["geometry"] = cells["geometry"].apply(
                    lambda geom: affinity.scale(geom, xfact=(1/scale_factor), yfact=(1/scale_factor), origin=(0, 0)))

            shapes = [(geom, idx) for idx, geom in enumerate(cells.geometry)]

            # Set the mask shape to be the same as the transcript bounds in rakaia
            mask = rasterize(shapes=shapes, out_shape=(int(y_max - y_min), int(x_max - x_min)),
                             transform=transform, dtype='int32')
            return np.flip(mask, axis=0) if flip else mask
        return None

    @staticmethod
    def expr_uses_scalefactors(shape_frame: pd.DataFrame,
                               scale_col: str='radius'):
        """
        Determine if a provided shape frame supports scale factors for spots i.e. 10x Visium

        :param shape_frame: `pd.DataFrame` of shape values associated with a particular ROI
        :param scale_col: Column identifier to match the shape frame identifier to the table geometry

        :return: Scaling float if the dataset uses a scaling factor, or `None` otherwise
        """
        if scale_col in shape_frame.columns and len(shape_frame[scale_col].unique()) == 1:
            return float(shape_frame[scale_col].unique())
        return None

    def _iterate_shapes_by_region(self, sdset: sd.SpatialData,
                                  id_col: str='region', scale_col: str='radius'):
        """
        Iterate each shape in spatialdata, treating it as a region with matched expr in `adata.obs['region']`

        :param sdset: `spatialdata` object containing either ST or multiplexed imaging measurements
        :param id_col: Identifier column in the table to link the shape identifier to expression profiles
        :param scale_col: Column identifier in the shape frame to identify geometry, such as scaling factors

        :return: Boolean indicating if any matched expression profiles by shape are found
        """
        found_expr = False
        if len(sdset.shapes) > 0:
            expr = sdset.tables['table']
            for shape in sdset.shapes:
                frame = pd.DataFrame(sdset[shape])
                sub_expr = expr[expr.obs[id_col] == str(shape)].copy() if id_col in expr.obs else expr
                if self.expr_uses_scalefactors(frame, scale_col):
                    rad = float(frame[scale_col].unique())
                    # only applies to 10X Visium spot-based
                    sub_expr.uns = {'spatial': {str(shape):
                                {'scalefactors': {'spot_diameter_fullres': 2 * rad}}}}
                if len(sub_expr) > 0 and is_spatial_dataset(sub_expr):
                    self._image_paths['uploads'].append(self.write_adata(sub_expr, shape))
                    found_expr = True
        return found_expr

    def _iterate_visium_hd_bins(self, sdata: sd.SpatialData,
                                output_masks: bool=True):
        """
        Iterate through matched shape frames and expression tables y bin size for Visium HD

        :param sdata: `Spatialdata` object containing a Visium HD dataset

        :return: None
        """
        # here, zip through the shapes and table to get the expr with a sample name
        for shape, table in zip(sdata.shapes, sdata.tables):
            # parse through the bin sizes, and see if it's in the current shape
            for bin_size in ZarrSDKeys.visium_hd_bin_sizes:
                if bin_size in table:
                    expr = self.scale_visium_hd_by_bin(sdata.tables[table], int(bin_size))
                    self._image_paths['uploads'].append(self.write_adata(expr, str(shape)))
                    if str(table) in str(shape) and output_masks:
                        mask = self.spatial_segmentation_mask(sdata, True, str(shape), str(table), int(bin_size))
                        if mask is not None:
                            self._mask_paths[str(shape)] = self.write_mask(mask, str(shape))

    def _parse(self, zarr_path: Union[Path, str]):
        """
        Parse the zarr store to detect the technology and create the temporary files

        :param zarr_path: Path to a `spatialdata` `zarr` store

        :return: None
        """
        sdata = sd.read_zarr(zarr_path)
        # Case 1: if it's Xenium
        if self.is_xenium(sdata):
            # for now for xenium, just use the zarr basename as the mapping is one ROI per zarr
            sam_name = str(os.path.basename(zarr_path)).replace('.', '_')
            self._image_paths['uploads'].append(self.write_adata(sdata.tables['table'], sam_name))
            mask = self.spatial_segmentation_mask(sdata)
            if mask is not None:
                self._mask_paths[sam_name] = self.write_mask(mask, sam_name)

        # Case 2: if it's Visium HD
        elif self.is_visium_hd(sdata):
            self._iterate_visium_hd_bins(sdata)

        else:
            # Case 3: process Visium by iterating the shapes (one per ROI, the shape key is the sample)
            found_expr = self._iterate_shapes_by_region(sdata)
            if not found_expr:
                # Case 4: if not 10x, iterate each table here as its own spatial expression set instead of using a set table key
                for table_key in sdata.tables.keys():
                    if is_spatial_dataset(sdata.tables[table_key]):
                        # use the table key in the file name output
                        self._image_paths['uploads'].append(self.write_adata(sdata.tables[table_key],
                        f"{str(os.path.basename(zarr_path)).replace('.', '_')}_{table_key}"))

    def get_files(self):
        """
        Return the files from a zarr store parse. Should match the output tuples from `parse_steinbock_dir`

        :return: Tuple of outputs matching the outputs from a steinbock directory: image paths, error config, mask paths,
        quantification, UMAP coordinates, and scaling JSON
        """
        if self._zarr_path is not None:
            self._parse(self._zarr_path)
        return (self._image_paths,) + tuple([self.truthy_no_update(val) for val in
                                             (self._error, self._mask_paths, self._quant,
                                              self._umap, self._scaling)])

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
    except (KeyError, TypeError, IndexError):
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
                               as_mask: int=False,
                               array_store_type: str="float"):
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
    spot_values = spatial_marker_to_dense_flat(spot_values).astype(
                set_array_storage_type_from_config(array_store_type))

    suf_expr = spot_values > 0
    spot_values = spot_values[suf_expr]

    # Extract spatial coordinates from the 'spatial' key in .obsm
    if 'spatial' not in adata.obsm.keys():
        raise ValueError("Spatial coordinates not found in 'obsm' attribute.")

    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
    capture_size, spot_size, scale_factor = set_spatial_scale(adata, spot_size, downscale=downscale)
    spot_size = spot_size if not downscale else int(spot_size * scale_factor)

    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(adata, downscale=downscale)

    spatial_coords = np.array(adata.obsm['spatial'] * scale_factor).astype(np.float32)
    # switch y and x coordinates to match row-col orientation
    spatial_coords = spatial_coords[:, [1, 0]]
    spatial_coords = spatial_coords[suf_expr]

    # Shift coordinates so min becomes (0, 0)
    spatial_coords -= np.array([y_min, x_min])

    # max_yx = spatial_coords.max(axis=0)
    image_shape = (int(grid_height), int(grid_width))

    # precompute disk offsets
    rr_offset, cc_offset = skimage.draw.disk((0, 0), int(spot_size))
    disk_coords = np.stack([rr_offset, cc_offset], axis=1)

    # vectorized disk placement
    K = disk_coords.shape[0]
    expanded_coords = spatial_coords[:, None, :] + disk_coords[None, :, :]
    flat_coords = expanded_coords.reshape(-1, 2).astype(np.uint32)
    yy, xx = flat_coords[:, 0], flat_coords[:, 1]

    # Place expression values (overlap-safe)
    repeated_values = np.repeat(spot_values, K)

    # Filter out-of-bounds
    valid = ((yy >= 0) & (yy < image_shape[0]) &
            (xx >= 0) & (xx < image_shape[1]))

    yy, xx = yy[valid], xx[valid]

    values = repeated_values[valid]

    flat_indices = yy * image_shape[1] + xx
    flat_image = np.zeros(image_shape[0] * image_shape[1], dtype=np.float32)

    # Use np.maximum.at to preserve the max-overwrite behavior
    np.maximum.at(flat_image, flat_indices, values)

    # Reshape back to 2D
    return flat_image.reshape(image_shape).astype(np.float32)


def check_spatial_array_multi_channel(image_dict: dict, data_selection: str,
                                      adata: ad.AnnData, channel_list: list,
                                      spot_size: Union[int,float]=55,
                                      array_store_type: str="float"):
    """
    Check the current raw image dictionary for missing spatial arrays using the currently selected
    marker list (current blend). If markers are missing, add the expression arrays (non-sparse) to the dictionary
    """
    for selection in channel_list:
        if not image_dict[data_selection] or image_dict[data_selection][selection] is None:
            image_dict[data_selection][selection] = spatial_grid_single_marker(adata, selection, spot_size,
                                                    True, False, array_store_type)
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

def spatial_selection_can_transfer_coordinates(data_selection: str,
                              session_uploads: Union[list, dict],
                              delimiter: str="+++",
                              transformation_matrix: Union[str, np.array, None]=None,):
    """
    Detect if the current data selection is a spatial dataset that can perform coordinate
    transfer to a WSI (i.e. H & E). Iterates over the current
    list of session uploads, matching the data selection to the upload. Then, it determines
    if the file associated is a spatial dataset with scaling factors (Visium spot-based)
    """
    exp, slide, acq = split_string_at_pattern(data_selection, delimiter)
    for upload in session_uploads['uploads']:
        if (exp in upload and upload.endswith('h5ad') and
                (visium_has_scaling_factors(ad.read_h5ad(upload)) or
                 visium_has_bin_scaling(ad.read_h5ad(upload)) or
                 transformation_matrix is not None)):
            return True, upload
    return False, None

def visium_has_bin_scaling(adata: Union[str, ad.AnnData],
                           key: str="scaling_visium_hd"):
    """
    Defines if the anndata object has the key defined in the `uns` slot, providing
    the bin sizing factor for Visium HD
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    return key in list(adata.uns_keys())

def get_visium_bin_scaling(adata: Union[str, ad.AnnData],
                           key: str="scaling_visium_hd"):
    """
    Set the Visium bin scaling size for either HD or spot-based arrays. HD files should
    have the bin factor saved using the specified key in the `uns` slot. For all other
    Visium assays, no bin scaling is performed (1)
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    if visium_has_bin_scaling(adata, key):
        return int(adata.uns[key])
    return 1

def visium_coords_to_wsi_from_zoom(bounds: dict,
                                   adata: Union[ad.AnnData, str]):
    """
    Convert a series of zoom coordinates from 10X Visium (V1,V2) to a matched WSI (i.e. H & E).
    Assumes that the pixel coordinates for the original hires image are in the `spatial`
    `obsm` slot
    Returns a string of comma separated values: x_min, y_min, height, width
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    capture_size, spot_size, scale_factor = set_spatial_scale(adata)
    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(adata)
    bin_scale_size = get_visium_bin_scaling(adata)
    # get the coordinates relative to the canvas image
    spatial_filtered = adata.obsm['spatial'].copy()
    spatial_filtered[:, 0] = ((spatial_filtered[:, 0]) * scale_factor) - x_min
    spatial_filtered[:, 1] = ((spatial_filtered[:, 1]) * scale_factor) - y_min
    # filter to the spots inside the zoom area
    x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(bounds)
    mask = (spatial_filtered[:, 0] >= x_low) & (spatial_filtered[:, 0] <= x_high) & \
           (spatial_filtered[:, 1] >= y_low) & (spatial_filtered[:, 1] <= y_high)
    adata_filtered = adata[mask]
    x_min, y_min = np.min(adata_filtered.obsm['spatial'], axis=0)
    x_max, y_max = np.max(adata_filtered.obsm['spatial'], axis=0)
    height = int(y_max - y_min) * bin_scale_size
    width = int(x_max - x_min) * bin_scale_size
    return f"{x_min * bin_scale_size},{y_min * bin_scale_size},{width},{height}"

def trim_neg_val(val):
    """
    Trim a negative value to a very small value during pixel conversion
    """
    return 0.1 if val < 0 else val

def xenium_coords_to_wsi_from_zoom(bounds: dict,
                                   adata: Union[ad.AnnData, str],
                                   transformation_matrix: Union[np.ndarray, str],
                                   scaling_factor: float=0.21):
    """
    Convert a series of zoom coordinates from 10X Xenium to a matched WSI (i.e. H & E).
    Assumes that an affine transformation matrix with the last row being `[0 0 1]` is provided,
    and that the pixel coordinates for the original hires image are in the `spatial`
    `obsm` slot. Additionally requires a Xenium scaling factor which by default is given at the 0 series level
    described here: https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    Returns a string of comma separated values: x_min, y_min, height, width
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    transformation_matrix = np.array(pd.read_csv(transformation_matrix, header=None)) if not (
        isinstance(transformation_matrix, np.ndarray)) else transformation_matrix
    grid_width, grid_height, x_min, y_min = spatial_canvas_dimensions(adata)
    x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(bounds)

    # get the bounds for two opposite corners to get the full bound
    both_low = np.matmul(inv(transformation_matrix),
                np.array([trim_neg_val((x_low + x_min) /scaling_factor),
                          trim_neg_val((y_low + y_min) /scaling_factor), 1]))

    both_high = np.matmul(inv(transformation_matrix),
                np.array([trim_neg_val((x_high + x_min) /scaling_factor),
                          trim_neg_val((y_high + y_min) /scaling_factor), 1]))

    concat = np.delete(np.vstack((both_low, both_high)), -1, axis=1)
    out_x_min, out_y_min = np.min(concat, axis=0)
    out_x_max, out_y_max = np.max(concat, axis=0)
    height = int(out_y_max - out_y_min)
    width = int(out_x_max - out_x_min)
    return f"{out_x_min},{out_y_min},{width},{height}"

def anndata_obs_to_projection_frame(adata: Union[ad.AnnData, str]):
    """
    Output the `anndata.obs` slot as a data frame compatible with categorical projection
    """
    adata = ad.read_h5ad(adata) if not isinstance(adata, ad.AnnData) else adata
    metadata = adata.obs.copy()
    # give each element an object id that matches to a mask
    # !IMPORTANT!: only works when the object order descending matches the mask order (i.e. Visium and Xenium)
    metadata['object_id'] = range(1, (len(adata) + 1), 1)
    metadata.index = range(1, (len(adata) + 1), 1)
    return metadata.drop('cell_id', axis=1) if 'cell_id' in metadata.columns else metadata
