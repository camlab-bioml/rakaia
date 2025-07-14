"""Module containing utility functions for object-related files such as masks and
quantification frames. Provides utilities for sub-setting and name matching
"""

from typing import Union
import re
from pydantic import BaseModel
import pandas as pd
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np
from skimage.segmentation import find_boundaries
import numexpr as ne
import scipy
from rakaia.utils.pixel import (
    path_to_mask,
    split_string_at_pattern)

class QuantificationColumns(BaseModel):
    """
    Holds the default named columns that can come after the channel columns
    """
    identifiers: list = ['sample', 'description']
    positions: list = ['sample', 'cell_id', 'description']
    defaults: list = ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample', 'x_min', 'y_min', 'object_annotation_1',
                      'rakaia_cell_annotation', 'PhenoGraph_clusters', 'Labels']

class QuantificationFormatError(Exception):
    """
    Raise an exception if any of the inputs required for object quantification are malformed or missing.
    """

def set_columns_to_drop(measurements_csv=None):
    """
    Parse a measurement data frame and create a list of columns that should be dropped for channel expression computation.
    """
    if measurements_csv is None:
        return QuantificationColumns().defaults
    # drop every column from sample and after, as these don't represent channels
    indices = []
    cols = list(measurements_csv.columns)
    for column in QuantificationColumns().positions:
        try:
            indices.append(cols.index(column))
        except (KeyError, ValueError, IndexError):
            pass
    if not indices:
        return QuantificationColumns().defaults
    return cols[min(indices): len(measurements_csv.columns)]

def set_mandatory_columns(only_sample=True):
    """
    Set the mandatory column list for a measurements data frame
    """
    if only_sample:
        return ['sample', 'cell_id']
    return ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']

def get_min_max_values_from_zoom_box(coord_dict):
    """
    Parse a dictionary entry for a canvas zoom instance, and get the min and max coordinate positions
    for both the x and y-axis
    """
    if not all(elem in coord_dict.keys() for elem in
                    ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']):
        return None, None, None, None
    x_min = min(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
    x_max = max(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
    y_min = min(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
    y_max = max(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
    return x_min, x_max, y_min, y_max

def get_min_max_values_from_rect_box(coord_dict):
    """
    Parse a dictionary entry for rectangle shape drawn on a canvas, and get the min and max coordinate positions
    for both the x and y-axis
    """
    if not all(elem in coord_dict.keys() for elem in
                    ['x0', 'x1', 'y0', 'y1']):
        return None, None, None, None
    x_min = min(coord_dict['x0'], coord_dict['x1'])
    x_max = max(coord_dict['x0'], coord_dict['x1'])
    y_min = min(coord_dict['y0'], coord_dict['y1'])
    y_max = max(coord_dict['y0'], coord_dict['y1'])
    return x_min, x_max, y_min, y_max

def convert_mask_to_object_boundary(mask):
    """
    Generate a mask of object outlines corresponding to a segmentation mask of integer objects
    Returns an RGB mask where outlines are represented by 255, and all else are 0
    Note that this mask does not retain the spatial location of individual objects
    """
    boundaries = find_boundaries(mask, mode='inner', connectivity=1).astype(np.uint8)
    return ne.evaluate("255 * boundaries").astype(np.uint8)


def subset_measurements_frame_from_umap_coordinates(measurements, umap_frame, coordinates_dict,
                                                    normalized_values=None, umap_overlay: Union[dict, None]=None):
    """
    Subset measurements frame based on a range of UMAP coordinates in the x and y axes
    Expects that the length of both frames are equal
    """
    measurements_to_use = normalized_values if normalized_values is not None else measurements
    grouping = None
    if umap_overlay:
        measurements_to_use[str(list(umap_overlay.keys())[0])] = pd.Series(list(umap_overlay.values())[0])
        grouping = str(list(umap_overlay.keys())[0])
    if not all(elem in coordinates_dict for elem in ['xaxis.range[0]', 'xaxis.range[1]',
            'yaxis.range[0]', 'yaxis.range[1]']): return measurements_to_use, grouping
    if len(measurements) != len(umap_frame):
        umap_frame = umap_frame.iloc[measurements.index.values.tolist()]
    query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                             f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                             f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                             f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')
    subset = measurements_to_use.loc[query.index.tolist()]
    return subset, grouping

def populate_quantification_frame_column_from_umap_subsetting(measurements, umap_frame, coordinates_dict,
                                        annotation_column="object_annotation_1", annotation_value="Unassigned",
                                        default_annotation_value="Unassigned"):
    """
    Populate a new column in the quantification frame with a column annotation with a value as subset using the
    interactive UMAP
    this is similar but differs from the annotation from the canvas as the coordinates represent UMAP dimensions
    and not pixels from an image it the coordinates dict
    """
    try:
        umap_frame.columns = ['UMAP1', 'UMAP2']
        if len(measurements) != len(umap_frame):
            umap_frame = umap_frame.iloc[measurements.index.values.tolist()]
        if all(elem in coordinates_dict for elem in ['xaxis.range[0]','xaxis.range[1]',
                                                          'yaxis.range[0]', 'yaxis.range[1]']):
            query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                         f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                         f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                         f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')

            list_indices = query.index.tolist()

            if annotation_column not in measurements.columns:
                measurements[annotation_column] = default_annotation_value

            measurements[annotation_column] = np.where(measurements.index.isin(list_indices),
                                                   annotation_value, measurements[annotation_column])
    except KeyError: pass
    return measurements

def send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection,
                                           mask_toggle):
    """
    Send an alert if the mask selected does not match the dimensions of the current canvas ROI
    If a mismatch, clear the selected mask name to avoid confusion
    """
    if None not in (mask_dict, data_selection, upload_dict, mask_selection) and mask_toggle:
        try:
            first_image = list(upload_dict[data_selection].keys())[0]
            first_image = upload_dict[data_selection][first_image]
            if first_image is not None and (first_image.shape[0] != mask_dict[mask_selection]["array"].shape[0] or
                    first_image.shape[1] != mask_dict[mask_selection]["array"].shape[1]):
                if error_config is None:
                    error_config = {"error": None}
                error_config["error"] = "Warning: the current mask does not have " \
                                    "the same dimensions as the current ROI."
                return error_config, None
            raise PreventUpdate
        except KeyError as exc:
            raise PreventUpdate from exc
    raise PreventUpdate

def subset_measurements_by_object_graph_box(measurements, coordinates_dict):
    """
    Subset a measurements CSV by getting the min and max coordinates in both dimensions from the canvas
    The query assumes a bounding box for the region in question and that the objects are wholly contained within
    the region
    The coordinates_dict assumes the following keys: ['xaxis.range[0]', 'xaxis.range[1]',
    'yaxis.range[0]', 'yaxis.range[1]']
    """
    try:
        return measurements.query(
            f'x_min >= {min(coordinates_dict["xaxis.range[0]"], coordinates_dict["xaxis.range[1]"])} &'
            f'x_max <= {max(coordinates_dict["xaxis.range[0]"], coordinates_dict["xaxis.range[1]"])} &'
            f'y_min >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
            f'y_max <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')
    except pd.errors.UndefinedVariableError:
        return None

def populate_object_annotation_column_from_bounding_box(measurements, coord_dict=None,
                annotation_column="object_annotation_1", values_dict=None, obj_type=None,
                box_type="zoom", remove: bool=False, default_val: str="Unassigned"):
    """
    Populate an object annotation column in the measurements data frame using numpy conditional searching
    by coordinate bounding box
    """
    if coord_dict is None:
        coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}
    try:
        if annotation_column not in measurements.columns:
            measurements[annotation_column] = default_val
        if box_type == "zoom":
            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(values_dict)
        elif box_type == "rect":
            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(values_dict)
        else:
            raise KeyError

        # if the annotation is being removed/overwritten, replace the annotation with the default
        obj_type = obj_type if not remove else default_val

        measurements[annotation_column] = np.where((measurements[str(f"{coord_dict['x_min']}")] >=
                                                        float(x_min)) &
                                               (measurements[str(f"{coord_dict['x_max']}")] <=
                                                float(x_max)) &
                                               (measurements[str(f"{coord_dict['y_min']}")] >=
                                                float(y_min)) &
                                               (measurements[str(f"{coord_dict['y_max']}")] <=
                                                float(y_max)),
                                                        obj_type,
                                                    measurements[annotation_column])
    except (KeyError, TypeError):
        pass
    return measurements

def populate_obj_annotation_column_from_obj_id_list(measurements, obj_list,
                    annotation_column="object_annotation_1", obj_identifier="cell_id", obj_type=None,
                    sample_name=None, id_column='sample', remove: bool=False, default_val: str="Unassigned"):
    """
    Populate a object annotation column in the measurements data frame using numpy conditional searching
    with a list of object IDs
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = default_val

    try:
        obj_type = obj_type if not remove else default_val
        measurements[annotation_column] = np.where((measurements[obj_identifier].isin(obj_list)) &
                                               (measurements[id_column] == sample_name), obj_type,
                                               measurements[annotation_column])
    except KeyError: pass
    return measurements


def populate_obj_annotation_column_from_clickpoint(measurements, coord_dict=None,
        annotation_column="object_annotation_1", obj_identifier="cell_id", values_dict=None,
        obj_type=None, mask_toggle=True, mask_dict=None, mask_selection=None, sample=None,
        id_column='sample', remove: bool=False, default_val: str="Unassigned"):
    """
    Populate an object annotation column in the measurements data frame from a single xy coordinate clickpoint
    """
    try:
        if annotation_column not in measurements.columns:
            measurements[annotation_column] = default_val

        if coord_dict is None:
            coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

        x_coord = values_dict['points'][0]['x']
        y_coord = values_dict['points'][0]['y']

        obj_type = obj_type if not remove else default_val

        if mask_toggle and None not in (mask_dict, mask_selection) and len(mask_dict) > 0:

            # get the object ID at that position to match
            mask_used = mask_dict[mask_selection]['raw']
            obj_id = mask_used[y_coord, x_coord].astype(int)
            obj_id = int(obj_id[0]) if isinstance(obj_id, np.ndarray) else obj_id

            measurements[annotation_column] = np.where((measurements[obj_identifier]== obj_id) &
                                                   (measurements[id_column] == sample), obj_type,
                                                   measurements[annotation_column])
        else:
            measurements[annotation_column] = np.where((measurements[str(f"{coord_dict['x_min']}")] <=
                                                        float(x_coord)) &
                                               (measurements[str(f"{coord_dict['x_max']}")] >=
                                                float(x_coord)) &
                                               (measurements[str(f"{coord_dict['y_min']}")] <=
                                                float(y_coord)) &
                                               (measurements[str(f"{coord_dict['y_max']}")] >=
                                                float(y_coord)) &
                                                (measurements['sample'] == sample),
                                                        obj_type,
                                                    measurements[annotation_column])
        return measurements
    except (KeyError, AssertionError):
        pass
    return measurements

def process_mask_array_for_hovertemplate(mask_array):
    """
    Process a mask array with mask object IDs for the hover template. Steps include:
    - converting the array shape to 3D with (shape[0], shape[1], 1)
    - Coercing array to string
    - Replacing '0' with None. 0 entries indicate that there is no mask object ID present at the pixel
    """
    mask_array = mask_array.astype(str)
    mask_array[mask_array == '0.0'] = 'None'
    mask_array[mask_array == '0'] = 'None'
    return mask_array.reshape((mask_array.shape[0], mask_array.shape[1], 1))


def get_objs_in_svg_boundary_by_mask_percentage(mask_array, svgpath, threshold=0.85, use_partial: bool=True):
    """
    Derive a list of object IDs from a mask that are contained within the svg path based on a threshold
    For example, with a threshold of 0.85, one would include an object ID if 85% of the object's pixels are
    contained within the svg path
    Returns a dict with the object ID from the mask and its percentage
    """
    bool_inside = path_to_mask(svgpath, (mask_array.shape[0], mask_array.shape[1]))
    if len(mask_array.shape) > 2:
        mask_array = mask_array.reshape((mask_array.shape[0], mask_array.shape[1]))
    if use_partial:
        mask_subset = np.where(bool_inside == True, mask_array, 0)
        objects = np.unique(mask_array)
        objects = objects[objects != 0]
        inside_counts = scipy.ndimage.sum(mask_subset, labels=mask_subset, index=objects)
        total_counts = scipy.ndimage.sum(mask_array, labels=mask_array, index=objects)
        mapping = {}
        percentages = [a / b for a, b in zip(inside_counts, total_counts)]
        for obj_id, percent in zip(objects, percentages):
            if percent >= threshold:
                mapping[obj_id] = percent
        return mapping
    uniques, counts = np.unique(mask_array[bool_inside], return_counts=True)
    uniques = uniques[uniques != 0]
    return {obj: 100 for obj in list(uniques)}

def subset_measurements_by_point(measurements, x_coord, y_coord):
    """
    Subset a measurements CSV by using a single xy coordinate. Assumes that only one entry in the measurements
    query is possible
    """
    try:
        return measurements.query(f'x_min <= {x_coord} & x_max >= {x_coord} & '
                                  f'y_min <= {y_coord} & y_max >= {y_coord}')
    except pd.errors.UndefinedVariableError:
        return None

def validate_mask_shape_matches_image(mask, image):
    """
    Return a boolean indicating if a given mask has dimensions that are compatible with an image array
    """
    return (mask.shape[0] == image.shape[0]) and (mask.shape[1] == image.shape[1])


def greyscale_grid_array(array_shape, dim=100):
    """
    Generate a greyscale grid array defined as white lines with a black background, with the square dimensions
    given by the dim parameter
    """
    empty = np.zeros(array_shape)
    rows_num = int(empty.shape[0] / dim)
    cols_num = int(empty.shape[1] / dim)

    cols_spacing = [int(i) for i in np.linspace(0, (empty.shape[1] - 1), cols_num)]
    rows_spacing = [int(i) for i in np.linspace(0, (empty.shape[0] - 1), rows_num)]

    # only create the grid lines if the image is sufficiently large: 2x the dimension of the box or greater
    if empty.shape[0] >= (2 * dim):
        for row in rows_spacing:
            empty[row] = 255

    if empty.shape[1] >= (2 * dim):
        for col in cols_spacing:
            empty[:, col] = 255

    return np.array(Image.fromarray(empty).convert('RGB')).astype(np.uint8)

def match_steinbock_mask_name_to_mcd_roi(mask_name: str=None, roi_name: str=None,
                                         return_mask_name: bool=True):
    """
    Match a steinbock output mask name to mcd ROI name
    Example match: Patient1_003 corresponds to pos1_3_3 (Patient1.mcd is the filename and pos1_3_3 is the third ROI
    in the file, so the mask with the third index should match)
    """
    if mask_name and roi_name:
        re_mask = re.search(r'\d+$', mask_name)
        re_roi = re.search(r'\d+$', roi_name)
        if re_mask and re_roi and int(re_mask.group()) == int(re_roi.group()):
            return mask_name if return_mask_name else roi_name
        return None
    return None

def is_steinbock_intensity_anndata(adata):
    """
    Identify if the anndata object is generated from steinbock. Factors include:
    - each Index in `adata.obs` begins with 'Object' followed by the object id
    - An 'Image' parameter in obs that contains the tiff information
    """
    # for now, these parameter check work for both .txt and .mcd pipeline runs
    return 'Image' in adata.obs and \
        all('.tiff' in elem for elem in adata.obs['Image'].to_list()) and \
      all('Object' in elem for elem in adata.obs.index)

class ROIQuantificationMatch:
    """
    Parse the quantification sheet and current ROI name to identify the column name to use to match
    the current ROI to the quantification sheet. Options are either `description` or `sample`. Description is
    prioritized as the name of the ROI, and sample is the file name with a 1-indexed counter such as {file_name}_1.
    Additionally used for parsing multi ROI cluster uploads wither either `roi` or `description` columns as identifiers

    :param data_selection: String representation of the current ROI selection
    :param quantification_frame: tabular dataset of summarized intensity measurements per object
    :param dataset_options: List of string representations of imported session ROIs
    :param delimiter: string to split the data selection string into experiment/filename, slide, and ROI identifier
    :param mask_name: string name of the currently applied mask (if it exists)
    :param cols_check: List of column identifiers to check as descriptors (efault is just `description`)
    :return: None
    """
    def __init__(self, data_selection, quantification_frame, dataset_options,
                                           delimiter: str="+++", mask_name: str=None,
                                           cols_check: Union[list, None]=None):

        self.data_selection = data_selection
        self.quantification = pd.DataFrame(quantification_frame)
        self.dataset_options = dataset_options
        self.delimiter = delimiter
        self.mask = mask_name
        self._match = None
        self._quant_col = None

        exp, slide, acq = split_string_at_pattern(self.data_selection, pattern=self.delimiter)
        self._cols_descriptor = self._set_descriptor_columns(cols_check)
        for col in self._cols_descriptor:
            if col in self.quantification.columns:
                # this part applies to ROIs from mcd
                self.steinbock_pipeline_description(exp, acq, col)
                # use experiment name if coming from tiff
                self.filename_overlap(exp, col)

        # if the descriptor column doesn't produce a match, try `sample` as a backup
        if 'sample' in self.quantification.columns and not self._match:
            self.steinbock_pipeline_sample(acq)
            self.match_by_dataset_index(exp)

    @staticmethod
    def _set_descriptor_columns(cols_check: Union[list, None]=None):
        """
        Set the descriptor columns to parse for matching. These columns are used for both
        ROI querying and matching for multi ROI cluster uploads.

        :param cols_check: The list of columns to check (default is just `description`)
        :return: tuple of descriptor columns to check against the quantification frame
        """
        return set(cols_check + ['description']) if cols_check else ['description']


    def steinbock_pipeline_description(self, experiment_identifier: str, roi_identifier: str,
                                       col_use: str='description'):
        """
        Match the quantification column to a mask based on the steinbock pipeline naming w/ description
        :param experiment_identifier: string experiment/filename identifier for the current ROI
        :param roi_identifier: string ROI identifier for the current ROI
        :param col_use: Name of the column to use as the ROI identifier (default is `description`)
        :return: None
        """
        if experiment_identifier and roi_identifier and not self._match:
            # important: if the index is less than 3 digits, i.e. 100 or more, pad with 0
            roi_index = roi_identifier.split('_')[-1]
            roi_index = pad_steinbock_roi_index(roi_index)
            pattern = f"{experiment_identifier}_{roi_index}"
            if pattern in self.quantification[col_use].tolist():
                self._match = pattern
                self._quant_col = col_use
        if not self._match and (self.mask and (match_steinbock_mask_name_to_mcd_roi(self.mask, roi_identifier) or
            roi_identifier in self.mask)) or roi_identifier in self.quantification[col_use].tolist():
            self._match = roi_identifier
            self._quant_col = col_use

    def steinbock_pipeline_sample(self, roi_identifier: str):
        """
        Match the quantification column to a mask based on the steinbock pipeline naming w/ sample
        :param roi_identifier: string ROI identifier for the current ROI
        :return: None
        """
        if not self._match and self.mask and match_steinbock_mask_name_to_mcd_roi(self.mask, roi_identifier) or \
                (self.mask == roi_identifier):
            self._match = self.mask
            self._quant_col = 'sample'

    def filename_overlap(self, experiment_name: str,
                         col_use: str='description'):
        """
        Match the quantification using either the experiment of ROI identifier w/ description

        :param experiment_name: string of the experiment/filename of the current ROI
        :param col_use: Name of the column to use as the ROI identifier (default is `description`)
        :return: None
        """
        if not self._match and (experiment_name and (experiment_name in
                                                     self.quantification[col_use].tolist()) or
                                (self.mask and experiment_name in self.mask)):
            self._match = self.mask if (self.mask and experiment_name in self.mask) else experiment_name
            self._quant_col = col_use

    def match_by_dataset_index(self, experiment_name: str):
        """
        Match the quantification and ROI using the old pipeline syntax (positional ROI indexing)

        :param experiment_name: string of the experiment/filename of the current ROI
        :return: None
        """
        if not self._match:
            try:
                index = self.dataset_options.index(self.data_selection) + 1
                # this is the default format coming out of the pipeline, but it doesn't always link the mask, ROI, and
                # quant sheet properly
                self._match = f"{experiment_name}_{index}"
                # sample_name = exp
                self._quant_col = 'sample'
            except (IndexError, ValueError, AttributeError):
                pass

    def get_matches(self):
        """

        :return: Tuple: string match for the roi identifier (or None), and the identifying column in the measurements.
        """
        return self._match, self._quant_col

def pad_steinbock_roi_index(roi_index: Union[int, str, None]=None):
    """
    Pad the steinbock ROI index to match the syntax of the mask name. Returns a string
    Example: 1 becomes 001 to match file_001
    12 becomes 012 to match file_012
    """
    roi_index = str(roi_index) if roi_index is not None else ""
    while len(roi_index) < 3:
        roi_index = f"0{roi_index}" if len(roi_index) < 3 else roi_index
    return str(roi_index)

def hex_series_to_rgb_array(hex_series: Union[list, pd.Series]):
    """Convert a `Pandas` series of hex strings to an (N, 3) uint8 RGB array for mapping."""
    return np.stack(hex_series.str.lstrip('#').apply(
        lambda x: tuple(int(x[i:i+2], 16) for i in (0, 2, 4))
    )).astype(np.uint8)

def mask_with_cluster_annotations(mask_array: np.array, cluster_frame: pd.DataFrame, cluster_annotations: dict,
                                  cluster_col: str = "cluster", obj_id_col: str = "cell_id", retain_objs=True,
                                  use_gating_subset: bool = False, gating_subset_list: Union[list, None]=None,
                                  cluster_option_subset: Union[list, None]=None,
                                  default_color: str="#000000"):
    """
    Generate a mask where cluster annotations are filled in with a specified colour. Incorporates
    both object ID and category sub-setting
    Returns a mask in RGB format
    """
    cluster_frame = pd.DataFrame(cluster_frame)
    cluster_frame = cluster_frame.astype({obj_id_col: np.int32, cluster_col: str})
    mask_array = mask_array.astype(np.uint32)
    try:
        if use_gating_subset:
            mask_bool = np.isin(mask_array, gating_subset_list)
            mask_array[~mask_bool] = 0
            mask_array = mask_array.astype(np.uint32)

        # Safely map group names to hex codes, using default color if group not in dict
        assign_frame = cluster_frame.copy()
        assign_frame = assign_frame[assign_frame[obj_id_col].isin(mask_array.flatten())]
        if cluster_option_subset is not None:
            assign_frame = assign_frame[assign_frame[cluster_col].isin(cluster_option_subset)]
        map_max = set_color_map_max(mask_array, pd.Series(assign_frame[obj_id_col])) + 1
        color_map = np.zeros((map_max, 3), dtype=np.uint8)
        assign_frame['color'] = assign_frame[cluster_col].map(cluster_annotations).fillna(default_color)

        # Convert hex codes to RGB
        rgb_colors = hex_series_to_rgb_array(assign_frame['color'])

        # Create a color lookup table: object_id → RGB
        object_ids = assign_frame[obj_id_col].to_numpy().astype(np.uint32)
        color_map[object_ids] = rgb_colors

        # Map full mask to RGB using the color map
        rgb_image = color_map[mask_array]
        return rgb_image
    except (KeyError, ValueError, IndexError):
        return None

def set_color_map_max(mask: np.array, id_vector: Union[list, pd.Series]):
    """
    Define the maximum value for the RGB mask color map. The color map should have enough values
    for all object ids in the mask to fill
    """
    return max(int(np.max(mask)), int(max(id_vector.to_list())))


def remove_annotation_entry_by_indices(annotations_dict: dict=None, roi_selection: str=None,
                                       index_list: list=None):
    """
    Remove annotation hash entries by a list of indices, generated either from the annotation preview
    table or triggering the list
    """
    if annotations_dict and roi_selection:
        annot_dict = annotations_dict.copy()
        key_list = list(annot_dict[roi_selection].keys())
        index_list = index_list if index_list else [-1]
        try:
            for index_to_remove in index_list:
                del annot_dict[roi_selection][key_list[index_to_remove]]
        except (KeyError, IndexError):
            pass
        return annot_dict
    return annotations_dict


def umap_fig_using_zoom(umap_layout: Union[dict, None]=None):
    """
    Define if the umap figure is currently using a zoom feature for object id sub-setting
    """
    return umap_layout is not None and all(elem in umap_layout for elem in
    ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]'])


def quantification_distribution_table(quantification_dict: Union[dict, pd.DataFrame],
                                      umap_variable: str, subset_cur_cat: Union[dict, None]=None,
                                      counts_col: str="Counts", proportion_col: str="Proportion",
                                      counts_threshold: Union[int, float]=100):
    """
    Compute the proportion of frequency counts for a UMAP distribution
    """
    if subset_cur_cat is None:
        frame = pd.DataFrame(quantification_dict)
        frame[umap_variable] = frame[umap_variable].apply(str)
        frame = frame[umap_variable].value_counts().reset_index().rename(
            columns={str(umap_variable): "Value", 'count': "Counts"})
    else:
        frame = pd.DataFrame(zip(list(subset_cur_cat.keys()), list(subset_cur_cat.values())),
                             columns=["Value", "Counts"])
    if len(frame) <= counts_threshold:
        frame[proportion_col] = frame[counts_col] / (frame[counts_col].abs().sum())
        return frame.round(3).sort_values(by=['Counts'], ascending=False).to_dict(orient="records")
    return [{"Value": f"NA ({umap_variable} > {counts_threshold} unique values)",
                "Counts": "NA", "Proportion": "NA"}]

def custom_gating_id_list(input_string: str=None):
    """
    Split a text input string using commas to retrieve a list of mask object IDs for gating
    Each id element must be compatible as an integer
    """
    if input_string:
        gating_list = []
        for elem in input_string.split(","):
            # ignore non integers
            try:
                gating_list.append(int(elem.strip()))
            except ValueError:
                pass
        return gating_list
    return []


def compute_image_similarity_from_overlay(quantification: Union[dict, pd.DataFrame],
                                          overlay: str):
    """
    Compute the inner product similarity for a series of ROIs as defined by their
    proportions using a UMAP overlay. Common overlays could include `leiden` or `phenograph` clustering
    Generates a nxn data frame matrix for n images with similarity scores. Higher scores indicate
    greater similarity between two images based on their cluster proportions
    """
    quantification = pd.DataFrame(quantification)
    image_id_col = "sample" if "sample" in list(quantification.columns) else "description"
    quantification[overlay] = quantification[overlay].apply(str)

    # number of overlay types to compare
    num_variables = len(quantification[overlay].value_counts())
    num_images = len(quantification[image_id_col].value_counts())
    matrix_cor = np.zeros((num_variables, num_images))
    samples = [str(i) for i in quantification[image_id_col].value_counts().index]
    unique_clusters = [str(i) for i in quantification[overlay].value_counts().index]
    index = 0
    for roi in samples:
        sub = quantification[quantification[image_id_col] == roi]
        type_dict = sub[overlay].value_counts().sort_index().to_dict()
        # make sure that the value counts has every element in the unique clusters, otherwise pad with missing cats
        for unique in unique_clusters:
            if unique not in type_dict.keys():
                type_dict[unique] = 0
        series = pd.Series(type_dict).sort_index()
        prop = [round(float(int(i) / len(sub)), 3) for i in series.to_list()]
        matrix_cor[:, index] = np.array(prop).flatten()
        index += 1

    # get the pairwise dot products for every observation
    return pd.DataFrame(np.dot(matrix_cor.T, matrix_cor), columns=samples, index=samples)

def find_similar_images(image_cor: Union[dict, pd.DataFrame, None], current_image_id: str,
                        num_query: int=3, identifier: str="sample"):
    """
    Parse a data frame of image similarity scores and pull out the top n similar images by score
    Return either a dictionary of indices or names depending on the format of the matching quantification sheet
    """
    try:
        image_cor = pd.DataFrame(image_cor)
        similar = []
        possible = list(image_cor[current_image_id].sort_values(ascending=False).index)
        ordered = sorted(possible)
        for image in possible:
            if str(image) != str(current_image_id):
                similar.append(ordered.index(image) if identifier == "sample" else image)
        similar = similar[0:num_query] if len(similar) > num_query else similar
        # support for both pipelines: old pipeline using sample needs indices, processing using steinbock uses
        # the description column and supports names
        return {"indices": similar} if identifier == "sample" else {"names": similar}
    except KeyError:
        return None
