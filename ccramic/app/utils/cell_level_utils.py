import dash
from dash_extensions.enrich import Serverside
from sklearn.preprocessing import StandardScaler
import sys
from ..utils.pixel_level_utils import *
from dash.exceptions import PreventUpdate

def set_columns_to_drop():
    return ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample', 'x_min', 'y_min', 'ccramic_cell_annotation']

def set_mandatory_columns():
    return ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']

def get_pixel(mask, i, j):
    if len(mask.shape) > 2:
        return mask[i][j][0]
    else:
        return mask[i][j]

def get_min_max_values_from_zoom_box(coord_dict):
    try:
        assert all([elem in coord_dict.keys() for elem in \
                    ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']])
        x_min = min(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
        x_max = max(coord_dict['xaxis.range[0]'], coord_dict['xaxis.range[1]'])
        y_min = min(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
        y_max = max(coord_dict['yaxis.range[0]'], coord_dict['yaxis.range[1]'])
        return x_min, x_max, y_min, y_max
    except AssertionError:
        return None

def convert_mask_to_cell_boundary(mask, outline_color=255, greyscale=True):
    """
    Convert a mask array with filled in cell masks to an array with drawn boundaries with black interiors of cells
    """
    if greyscale:
        outlines = np.full((mask.shape[0], mask.shape[1]), 3)
    else:
        outlines = np.stack([np.empty(mask[0].shape), mask[0], mask[1]], axis=2)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            pixel = get_pixel(mask, i, j)
            if pixel != 0:
                if i != 0 and get_pixel(mask, i - 1, j) != pixel:
                    outlines[i][j] = outline_color
                elif i != mask.shape[0] - 1 and get_pixel(mask, i + 1, j) != pixel:
                    outlines[i][j] = outline_color
                elif j != 0 and get_pixel(mask, i, j - 1) != pixel:
                    outlines[i][j] = outline_color
                elif j != mask.shape[1] - 1 and get_pixel(mask, i, j + 1) != pixel:
                    outlines[i][j] = outline_color

    # Floating point errors can occaisionally put us very slightly below 0
    return np.where(outlines >= 0, outlines, 0).astype(np.uint8)


def subset_measurements_frame_from_umap_coordinates(measurements, umap_frame, coordinates_dict):
    """
    Subset measurements frame based on a range of UMAP coordinates in the x and y axes
    Expects that the length of both frames are equal
    """
    try:
        assert all([elem in coordinates_dict for elem in ['xaxis.range[0]','xaxis.range[1]',
                                                          'yaxis.range[0]', 'yaxis.range[1]']])
        query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                         f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                         f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                         f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')
        return measurements.loc[umap_frame.index[query.index.tolist()]]
    except AssertionError:
        return None

def send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection,
                                           mask_toggle):
    if None not in (mask_dict, data_selection, upload_dict, mask_selection) and mask_toggle:
        split = split_string_at_pattern(data_selection)
        exp, slide, acq = split[0], split[1], split[2]
        first_image = list(upload_dict[exp][slide][acq].keys())[0]
        first_image = upload_dict[exp][slide][acq][first_image]
        if first_image.shape[0] != mask_dict[mask_selection]["array"].shape[0] or \
                first_image.shape[1] != mask_dict[mask_selection]["array"].shape[1]:
            if error_config is None:
                error_config = {"error": None}
            error_config["error"] = "Warning: the current mask does not have " \
                                    "the same dimensions as the current ROI."
            return error_config
        else:
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
                                                    annotation_column="ccramic_cell_annotation",
                                                    values_dict=None,
                                                    cell_type=None):
    """
    Populate a cell annotation column in the measurements data frame using numpy conditional searching
    by coordinate bounding box
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = "None"

    if coord_dict is None:
        coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

    try:
        x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(values_dict)
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
    except (KeyError, AssertionError):
        pass

    return measurements
