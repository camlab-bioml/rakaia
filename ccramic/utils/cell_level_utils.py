from typing import Union
import pandas as pd
import re
from ccramic.utils.pixel_level_utils import (
    path_to_mask,
    split_string_at_pattern,
    recolour_greyscale)
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np
from skimage.segmentation import find_boundaries
from pydantic import BaseModel
import scipy

class QuantificationColumns(BaseModel):
    """
    Holds the default named columns that can come after the channel columns
    """
    identifiers: list = ['sample', 'description']
    positions: list = ['sample', 'cell_id', 'description']
    defaults: list = ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample', 'x_min', 'y_min', 'ccramic_cell_annotation',
                        'PhenoGraph_clusters', 'Labels']

class QuantificationFormatError(Exception):
    pass

def set_columns_to_drop(measurements_csv=None):
    if measurements_csv is None:
        return QuantificationColumns().defaults
    else:
        # drop every column from sample and after, as these don't represent channels
        indices = []
        cols = list(measurements_csv.columns)
        for column in QuantificationColumns().positions:
            try:
                indices.append(cols.index(column))
            except (KeyError, ValueError, IndexError):
                pass
        if not indices:
            # TODO: decide if throw error for quantification results that are missing the key identifying columns
            return QuantificationColumns().defaults
        return cols[min(indices): len(measurements_csv.columns)]

def set_mandatory_columns(only_sample=True):
    if only_sample:
        return ['sample', 'cell_id']
    else:
        return ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']

def get_min_max_values_from_zoom_box(coord_dict):
    if not all([elem in coord_dict.keys() for elem in \
                    ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']]): return None
    x_min = min(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
    x_max = max(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
    y_min = min(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
    y_max = max(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
    return x_min, x_max, y_min, y_max

def get_min_max_values_from_rect_box(coord_dict):
    if not all([elem in coord_dict.keys() for elem in \
                    ['x0', 'x1', 'y0', 'y1']]): return None
    x_min = min(coord_dict['x0'], coord_dict['x1'])
    x_max = max(coord_dict['x0'], coord_dict['x1'])
    y_min = min(coord_dict['y0'], coord_dict['y1'])
    y_max = max(coord_dict['y0'], coord_dict['y1'])
    return x_min, x_max, y_min, y_max

def convert_mask_to_cell_boundary(mask):
    boundaries = find_boundaries(mask, mode='inner', connectivity=1)
    return np.where(boundaries == True, 255, 0).astype(np.uint8)


def subset_measurements_frame_from_umap_coordinates(measurements, umap_frame, coordinates_dict, normalized_values=None):
    """
    Subset measurements frame based on a range of UMAP coordinates in the x and y axes
    Expects that the length of both frames are equal
    """
    if not all([elem in coordinates_dict for elem in ['xaxis.range[0]', 'xaxis.range[1]',
                                                      'yaxis.range[0]', 'yaxis.range[1]']]): return None
    if len(measurements) != len(umap_frame):
        umap_frame = umap_frame.iloc[measurements.index.values.tolist()]
    #     umap_frame.reset_index()
    #     measurements.reset_index()
    query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                             f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                             f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                             f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')
    # if len(measurements) != len(umap_frame):
    #     query.reset_index()
    # use the normalized values if they exist
    measurements_to_use = normalized_values if normalized_values is not None else measurements
    subset = measurements_to_use.loc[query.index.tolist()]
    return subset


def populate_quantification_frame_column_from_umap_subsetting(measurements, umap_frame, coordinates_dict,
                                        annotation_column="ccramic_cell_annotation", annotation_value="Unassigned",
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
        #     umap_frame.reset_index()
        #     measurements.reset_index()
        if all([elem in coordinates_dict for elem in ['xaxis.range[0]','xaxis.range[1]',
                                                          'yaxis.range[0]', 'yaxis.range[1]']]):
            query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                         f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                         f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                         f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')

            list_indices = query.index.tolist()

            if annotation_column not in measurements.columns:
                measurements[annotation_column] = default_annotation_value

            measurements[annotation_column] = np.where(measurements.index.isin(list_indices),
                                                   annotation_value, measurements[annotation_column])
    except KeyError:
        pass
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
            if first_image.shape[0] != mask_dict[mask_selection]["array"].shape[0] or \
                    first_image.shape[1] != mask_dict[mask_selection]["array"].shape[1]:
                if error_config is None:
                    error_config = {"error": None}
                error_config["error"] = "Warning: the current mask does not have " \
                                    "the same dimensions as the current ROI."
                return error_config, None
            else:
                raise PreventUpdate
        except KeyError:
            raise PreventUpdate
    else:
        raise PreventUpdate

def subset_measurements_by_cell_graph_box(measurements, coordinates_dict):
    """
    Subset a measurements CSV by getting the min and max coordinates in both dimensions from the canvas
    The query assumes a bounding box for the region in question and that the cells are wholly contained within
    the region
    The coordinates_dict assumes the following keys: ['xaxis.range[0]', 'xaxis.range[1]',
    'yaxis.range[0]', 'yaxis.range[1]']
    """
    #TODO: convert the query into a numpy where statement to fill in the cell type annotation in a new column
    # while preserving the existing data frame structure
    try:
        return measurements.query(
            f'x_min >= {min(coordinates_dict["xaxis.range[0]"], coordinates_dict["xaxis.range[1]"])} &'
            f'x_max <= {max(coordinates_dict["xaxis.range[0]"], coordinates_dict["xaxis.range[1]"])} &'
            f'y_min >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
            f'y_max <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')
    except pd.errors.UndefinedVariableError:
        return None

def populate_cell_annotation_column_from_bounding_box(measurements, coord_dict=None,
                        annotation_column="ccramic_cell_annotation", values_dict=None, cell_type=None,
                        box_type="zoom", remove: bool=False, default_val: str="Unassigned"):
    """
    Populate a cell annotation column in the measurements data frame using numpy conditional searching
    by coordinate bounding box
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = default_val

    if coord_dict is None:
        coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

    try:
        if box_type == "zoom":
            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(values_dict)
        elif box_type == "rect":
            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(values_dict)
        else:
            raise KeyError

        # if the annotation is being removed/overwritten, replace the annotation with the default
        cell_type = cell_type if not remove else default_val

        measurements[annotation_column] = np.where((measurements[str(f"{coord_dict['x_min']}")] >=
                                                        float(x_min)) &
                                               (measurements[str(f"{coord_dict['x_max']}")] <=
                                                float(x_max)) &
                                               (measurements[str(f"{coord_dict['y_min']}")] >=
                                                float(y_min)) &
                                               (measurements[str(f"{coord_dict['y_max']}")] <=
                                                float(y_max)),
                                                        cell_type,
                                                    measurements[annotation_column])
    except KeyError:
        pass

    return measurements

def populate_cell_annotation_column_from_cell_id_list(measurements, cell_list,
                        annotation_column="ccramic_cell_annotation", cell_identifier="cell_id", cell_type=None,
                        sample_name=None, id_column='sample', remove: bool=False, default_val: str="Unassigned"):
    """
    Populate a cell annotation column in the measurements data frame using numpy conditional searching
    with a list of cell IDs
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = default_val

    try:
        cell_type = cell_type if not remove else default_val
        measurements[annotation_column] = np.where((measurements[cell_identifier].isin(cell_list)) &
                                               (measurements[id_column] == sample_name), cell_type,
                                               measurements[annotation_column])
    except KeyError:
        pass
    return measurements


def populate_cell_annotation_column_from_clickpoint(measurements, coord_dict=None,
                    annotation_column="ccramic_cell_annotation", cell_identifier="cell_id", values_dict=None,
                    cell_type=None, mask_toggle=True, mask_dict=None, mask_selection=None, sample=None,
                    id_column='sample', remove: bool=False, default_val: str="Unassigned"):
    """
    Populate a cell annotation column in the measurements data frame from a single xy coordinate clickpoint
    """
    try:
        if annotation_column not in measurements.columns:
            measurements[annotation_column] = default_val

        if coord_dict is None:
            coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

        x = values_dict['points'][0]['x']
        y = values_dict['points'][0]['y']

        cell_type = cell_type if not remove else default_val

        if mask_toggle and None not in (mask_dict, mask_selection) and len(mask_dict) > 0:

            # get the cell ID at that position to match
            mask_used = mask_dict[mask_selection]['raw']
            cell_id = mask_used[y, x].astype(int)
            cell_id = int(cell_id[0]) if isinstance(cell_id, np.ndarray) else cell_id

            measurements[annotation_column] = np.where((measurements[cell_identifier]== cell_id) &
                                                   (measurements[id_column] == sample), cell_type,
                                                   measurements[annotation_column])
        else:
            measurements[annotation_column] = np.where((measurements[str(f"{coord_dict['x_min']}")] <=
                                                        float(x)) &
                                               (measurements[str(f"{coord_dict['x_max']}")] >=
                                                float(x)) &
                                               (measurements[str(f"{coord_dict['y_min']}")] <=
                                                float(y)) &
                                               (measurements[str(f"{coord_dict['y_max']}")] >=
                                                float(y)) &
                                                (measurements['sample'] == sample),
                                                        cell_type,
                                                    measurements[annotation_column])
        return measurements
    except (KeyError, AssertionError):
        pass
    return measurements

def process_mask_array_for_hovertemplate(mask_array):
    """
    Process a mask array with cell IDs for the hover template. Steps include:
    - converting the array shape to 3D with (shape[0], shape[1], 1)
    - Coercing array to string
    - Replacing '0' with None. 0 entries indicate that there is no cell ID present at the pixel
    """
    mask_array = mask_array.astype(str)
    mask_array[mask_array == '0.0'] = 'None'
    mask_array[mask_array == '0'] = 'None'
    return mask_array.reshape((mask_array.shape[0], mask_array.shape[1], 1))


def get_cells_in_svg_boundary_by_mask_percentage(mask_array, svgpath, threshold=0.85, use_partial: bool=True):
    """
    Derive a list of cell IDs from a mask that are contained within the svg path based on a threshold
    For example, with a threshold of 0.85, one would include a cell ID if 85% of the cell's pixels are
    contained within the svg path
    Returns a dict with the cell ID from the mask and its percentage
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
    return {cell: 100 for cell in list(uniques)}

def subset_measurements_by_point(measurements, x, y):
    """
    Subset a measurements CSV by using a single xy coordinate. Assumes that only one entry in the measurements
    query is possible
    """
    #TODO: convert the query into a numpy where statement to fill in the cell type annotation in a new column
    # while preserving the existing data frame structure
    try:
        return measurements.query(f'x_min <= {x} & x_max >= {x} & y_min <= {y} & y_max >= {y}')
    except pd.errors.UndefinedVariableError:
        return None

def validate_mask_shape_matches_image(mask, image):
    return (mask.shape[0] == image.shape[0]) and (mask.shape[1] == image.shape[1])


def generate_greyscale_grid_array(array_shape, dim=100):
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
    Match a steinbock output mask name to an mcd ROI name
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
    - each Index in adata.obs begins with 'Object' followed by the object id
    - An 'Image' parameter in obs that contains the tiff information
    """
    return 'Image' in adata.obs and 'image_acquisition_description' in adata.obs and \
        all(['.tiff' in elem for elem in adata.obs['Image'].to_list()]) and \
      all(['Object' in elem for elem in adata.obs.index])

def identify_column_matching_roi_to_quantification(data_selection, quantification_frame, dataset_options,
                                                   delimiter: str="+++", mask_name: str=None):
    """
    Parse the quantification sheet and current ROI name to identify the column name to use to match
    the current ROI to the quantification sheet. Options are either `description` or `sample`. Description is
    prioritized as the name of the ROI, and sample is the file name with a 1-indexed counter such as {file_name}_1
    """
    quantification_frame = pd.DataFrame(quantification_frame)
    exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
    if 'description' in quantification_frame.columns:
        # this part applies to ROIs from mcd
        if (mask_name and (match_steinbock_mask_name_to_mcd_roi(mask_name, acq) or acq in mask_name)) or \
                acq in quantification_frame['description'].tolist():
            return acq, 'description'
        # use experiment name if coming from tiff
        elif exp and (exp in quantification_frame['description'].tolist()) or (mask_name and exp in mask_name):
            return mask_name if exp in mask_name else exp, 'description'
        return None, None
    elif 'sample' in quantification_frame.columns:
        if mask_name and match_steinbock_mask_name_to_mcd_roi(mask_name, acq) or (mask_name == acq):
            return mask_name, 'sample'
        else:
            try:
                index = dataset_options.index(data_selection) + 1
                # this is the default format coming out of the pipeline, but it doesn't always link the mask, ROI, and
                # quant sheet properly
                sample_name = f"{exp}_{index}"
                # sample_name = exp
                return sample_name, 'sample'
            except IndexError:
                return None, None
    else:
        return None, None

def generate_mask_with_cluster_annotations(mask_array: np.array, cluster_frame: pd.DataFrame, cluster_annotations: dict,
                                           cluster_col: str = "cluster", cell_id_col: str = "cell_id", retain_cells=True,
                                           use_gating_subset: bool = False, gating_subset_list: list=None,
                                           cluster_option_subset=None):
    """
    Generate a mask where cluster annotations are filled in with a specified colour, and non-annotated cells
    remain as greyscale values
    Returns a mask in RGB format
    """
    cluster_frame = pd.DataFrame(cluster_frame)
    cluster_frame = cluster_frame.astype(str)
    empty = np.zeros((mask_array.shape[0], mask_array.shape[1], 3))
    mask_array = mask_array.astype(np.uint32)
    if use_gating_subset:
        mask_bool = np.isin(mask_array, gating_subset_list)
        mask_array[~mask_bool] = 0
    try:
        # set the cluster assignments to use either from the subset, or the default of all
        clusters_to_use = [str(select) for select in cluster_option_subset] if cluster_option_subset is not None else \
            cluster_frame[cluster_col].unique().tolist()
        for cell_type in clusters_to_use:
            cell_list = cluster_frame[(cluster_frame[str(cluster_col)] == str(cell_type))][cell_id_col].tolist()
            # make sure that the cells are integers so that they match the array values of the mask
            cell_list = [int(i) for i in cell_list]
            # TODO: try to speed up for very large images with either many cells or many cluster categories
            # time complexity is O(num_clusters)
            annot_mask = np.where(np.isin(mask_array, cell_list), mask_array, 0)
            annot_mask = np.where(annot_mask > 0, 255, 0).astype(np.float32)
            annot_mask = recolour_greyscale(annot_mask, cluster_annotations[cell_type])
            empty = empty + annot_mask
        # Find where the cells are annotated, and add back in the ones that are not
        if retain_cells:
            already_cells = np.array(Image.fromarray(empty.astype(np.uint8)).convert('L')) != 0
            mask_array[already_cells] = 0
            mask_to_add = np.array(Image.fromarray(mask_array).convert('RGB'))
            mask_to_add = np.where(mask_to_add > 0, 255, 0).astype(empty.dtype)
            return (empty + mask_to_add).clip(0, 255).astype(np.uint8)
        else:
            return empty.astype(np.uint8)
    except KeyError:
        return None

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

def quantification_distribution_table(quantification_dict: Union[dict, pd.DataFrame],
                                      umap_variable: str, subset_cur_cat: Union[dict, None]=None,
                                      counts_col: str="Counts", proportion_col: str="Proportion"):
    """
    Compute the proportion of frequency counts for a UMAP distribution
    """
    if subset_cur_cat is None:
        frame = pd.DataFrame(quantification_dict)[umap_variable].value_counts().reset_index().rename(
            columns={str(umap_variable): "Value", 'count': "Counts"})
    else:
        frame = pd.DataFrame(zip(list(subset_cur_cat.keys()), list(subset_cur_cat.values())),
                             columns=["Value", "Counts"])
    frame[proportion_col] = frame[counts_col] / (frame[counts_col].abs().sum())
    return frame.round(3).to_dict(orient="records")
