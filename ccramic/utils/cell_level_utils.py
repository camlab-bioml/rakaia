import numpy
import pandas as pd
from ccramic.utils.pixel_level_utils import (
    path_to_mask,
    get_bounding_box_for_svgpath,
    split_string_at_pattern,
    recolour_greyscale)
from dash.exceptions import PreventUpdate
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import os
import matplotlib.patches as mpatches
import numpy as np
from skimage.segmentation import find_boundaries


def set_columns_to_drop(measurements_csv=None):
    defaults = ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample', 'x_min', 'y_min', 'ccramic_cell_annotation',
            'PhenoGraph_clusters', 'Labels']
    if measurements_csv is None:
        return defaults
    else:
        # drop every column from sample and after, as these don't represent channels
        try:
            cols = list(measurements_csv.columns)
            index_find = min(cols.index('sample'), cols.index('cell_id'))
            return cols[index_find: len(cols)]
        except (ValueError, IndexError):
            return defaults

def set_mandatory_columns(only_sample=True):
    if only_sample:
        return ['sample', 'cell_id']
    else:
        return ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']

# def get_pixel(mask, i, j):
#     if len(mask.shape) > 2:
#         return mask[i][j][0]
#     else:
#         return mask[i][j]

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

def get_min_max_values_from_rect_box(coord_dict):
    try:
        assert all([elem in coord_dict.keys() for elem in \
                    ['x0', 'x1', 'y0', 'y1']])
        x_min = min(coord_dict['x0'], coord_dict['x1'])
        x_max = max(coord_dict['x0'], coord_dict['x1'])
        y_min = min(coord_dict['y0'], coord_dict['y1'])
        y_max = max(coord_dict['y0'], coord_dict['y1'])
        return x_min, x_max, y_min, y_max
    except AssertionError:
        return None

# def convert_mask_to_cell_boundary(mask, outline_color=255, greyscale=True):
#     """
#     Convert a mask array with filled in cell masks to an array with drawn boundaries with black interiors of cells
#     """
#     if greyscale:
#         outlines = np.full((mask.shape[0], mask.shape[1]), 3)
#     else:
#         outlines = np.stack([np.empty(mask[0].shape), mask[0], mask[1]], axis=2)
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             pixel = get_pixel(mask, i, j)
#             if pixel != 0:
#                 if i != 0 and get_pixel(mask, i - 1, j) != pixel:
#                     outlines[i][j] = outline_color
#                 elif i != mask.shape[0] - 1 and get_pixel(mask, i + 1, j) != pixel:
#                     outlines[i][j] = outline_color
#                 elif j != 0 and get_pixel(mask, i, j - 1) != pixel:
#                     outlines[i][j] = outline_color
#                 elif j != mask.shape[1] - 1 and get_pixel(mask, i, j + 1) != pixel:
#                     outlines[i][j] = outline_color
#
#     # Floating point errors can occaisionally put us very slightly below 0
#     return np.where(outlines >= 0, outlines, 0).astype(np.uint8)

def convert_mask_to_cell_boundary(mask):
    boundaries = find_boundaries(mask, mode='outer', connectivity=1)
    return np.where(boundaries == True, 255, 0).astype(np.uint8)


def subset_measurements_frame_from_umap_coordinates(measurements, umap_frame, coordinates_dict):
    """
    Subset measurements frame based on a range of UMAP coordinates in the x and y axes
    Expects that the length of both frames are equal
    """
    try:
        assert all([elem in coordinates_dict for elem in ['xaxis.range[0]','xaxis.range[1]',
                                                          'yaxis.range[0]', 'yaxis.range[1]']])
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
        subset = measurements.loc[query.index.tolist()]
        return subset
    except AssertionError:
        return None


def populate_quantification_frame_column_from_umap_subsetting(measurements, umap_frame, coordinates_dict,
                                        annotation_column="ccramic_cell_annotation", annotation_value="None"):
    """
    Populate a new column in the quantification frame with a column annotation with a value as subset using the
    interactive UMAP
    this is similar but differs from the annotation from the canvas as the coordinates represent UMAP dimensions
    and not pixels from an image it the coordinates dict
    """
    try:
        umap_frame.columns = ['UMAP1', 'UMAP2']
        assert all([elem in coordinates_dict for elem in ['xaxis.range[0]','xaxis.range[1]',
                                                          'yaxis.range[0]', 'yaxis.range[1]']])
        if len(measurements) != len(umap_frame):
            umap_frame = umap_frame.iloc[measurements.index.values.tolist()]
        #     umap_frame.reset_index()
        #     measurements.reset_index()
        query = umap_frame.query(f'UMAP1 >= {coordinates_dict["xaxis.range[0]"]} &'
                         f'UMAP1 <= {coordinates_dict["xaxis.range[1]"]} &'
                         f'UMAP2 >= {min(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])} &'
                         f'UMAP2 <= {max(coordinates_dict["yaxis.range[0]"], coordinates_dict["yaxis.range[1]"])}')

        list_indices = query.index.tolist()

        if annotation_column not in measurements.columns:
            measurements[annotation_column] = "None"

        measurements[annotation_column] = np.where(measurements.index.isin(list_indices),
                                                   annotation_value, measurements[annotation_column])

    except (KeyError, AssertionError):
        pass
    return measurements

def send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection,
                                           mask_toggle):
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
                return error_config
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
                                                    annotation_column="ccramic_cell_annotation",
                                                    values_dict=None,
                                                    cell_type=None, box_type="zoom"):
    """
    Populate a cell annotation column in the measurements data frame using numpy conditional searching
    by coordinate bounding box
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = "None"

    if coord_dict is None:
        coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

    try:
        if box_type == "zoom":
            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(values_dict)
        elif box_type == "rect":
            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(values_dict)
        else:
            raise KeyError
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

def populate_cell_annotation_column_from_cell_id_list(measurements, cell_list,
                                                    annotation_column="ccramic_cell_annotation",
                                                    cell_identifier="cell_id",
                                                    cell_type=None, sample_name=None, id_column='sample'):
    """
    Populate a cell annotation column in the measurements data frame using numpy conditional searching
    with a list of cell IDs
    """
    if annotation_column not in measurements.columns:
        measurements[annotation_column] = "None"

    try:
        measurements[annotation_column] = np.where((measurements[cell_identifier].isin(cell_list)) &
                                               (measurements[id_column] == sample_name), cell_type,
                                               measurements[annotation_column])
    except KeyError:
        pass
    return measurements


def populate_cell_annotation_column_from_clickpoint(measurements, coord_dict=None,
                                                    annotation_column="ccramic_cell_annotation",
                                                    cell_identifier="cell_id", values_dict=None, cell_type=None,
                                                    mask_toggle=True, mask_dict=None, mask_selection=None,
                                                    sample=None, id_column='sample'):
    """
    Populate a cell annotation column in the measurements data frame from a single xy coordinate clickpoint
    """
    try:
        if annotation_column not in measurements.columns:
            measurements[annotation_column] = "None"

        if coord_dict is None:
            coord_dict = {"x_min": "x_min", "x_max": "x_max", "y_min": "y_min", "y_max": "y_max"}

        x = values_dict['points'][0]['x']
        y = values_dict['points'][0]['y']

        if mask_toggle and None not in (mask_dict, mask_selection) and len(mask_dict) > 0:

            # get the cell ID at that position to match
            mask_used = mask_dict[mask_selection]['raw']
            cell_id = mask_used[y, x].astype(int)
            cell_id = int(cell_id[0]) if isinstance(cell_id, numpy.ndarray) else cell_id

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

def get_cells_in_svg_boundary_by_mask_percentage(mask_array, svgpath, threshold=0.85):
    """
    Derive a list of cell IDs from a mask that are contained within an svg path based on a threshold
    For example, with a threshold of 0.85, one would include a cell ID if 85% of the cell's pixels are
    contained within the svg path
    Returns a dict with the cell ID from the mask and its percentage
    """
    bool_inside = path_to_mask(svgpath, (mask_array.shape[0], mask_array.shape[1]))
    uniques, counts = np.unique(mask_array[bool_inside], return_counts=True)
    channel_index = 0
    cells_included = {}
    for cell in list(uniques):
        if int(cell) > 0:
            where_is_cell = np.where(mask_array == cell)
            percent = counts[channel_index] / len(mask_array[where_is_cell])
            if percent >= threshold:
                cells_included[cell] = percent
        channel_index += 1
    return cells_included

def generate_annotations_output_pdf(annotations_dict, canvas_layers, data_selection, mask_config,
                                    aliases, dest_dir="/tmp/", output_file="annotations.pdf", blend_dict=None):
    """
    Generate a PDF output report with region images linked to annotations.
    The annotations are held in a dictionary with the title, description, shapes/coordinates, and channels used
    Each annotation must be transformed into a region that is rendered as an image blend with the channels used
    """
    # subset = array[np.ix_(range(int(y_range_low), int(y_range_high), 1),
    #                       range(int(x_range_low), int(x_range_high), 1))]

    # ensure that the annotations are taken from the current ROI
    file_output = os.path.join(dest_dir, output_file)
    if data_selection in annotations_dict and len(annotations_dict) > 0:
        annotations_dict = {key: value for key, value in annotations_dict[data_selection].items() if \
                        value['type'] not in ['point']}
        if len(annotations_dict) > 0:
            with PdfPages(file_output) as pdf:
                for key, value in annotations_dict.items():
                    if value['type'] in ['zoom', 'rect', 'path']:
                        # the key is the tuple of the coordinates or the svgpath
                        if value['type'] == "zoom":
                            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
                        elif value['type'] == "rect":
                            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))
                        elif value['type'] == "path":
                            x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(key)
                        try:
                            image = sum([np.asarray(canvas_layers[data_selection][elem]).astype(np.float32) for \
                             elem in value['channels'] if \
                             elem in canvas_layers[data_selection].keys()]).astype(np.float32)
                            image = np.clip(image, 0, 255)
                        except KeyError:
                            image = None
                        if value['use_mask'] and None not in (mask_config, value['mask_selection']) and \
                                len(mask_config) > 0:
                            if image.shape[0] == mask_config[value['mask_selection']]["array"].shape[0] and \
                                image.shape[1] == mask_config[value['mask_selection']]["array"].shape[1]:
                                # set the mask blending level based on the slider, by default use an equal blend
                                mask_level = float(value['mask_blending_level'] / 100) if \
                                value['mask_blending_level'] is not None else 1
                                image = cv2.addWeighted(image.astype(np.uint8), 1,
                                                mask_config[value['mask_selection']]["array"].astype(np.uint8),
                                                mask_level, 0)
                            if value['add_mask_boundary'] and mask_config[value['mask_selection']]["boundary"] is not None:
                                    # add the border of the mask after converting back to greyscale to derive the conversion
                                reconverted = np.array(Image.fromarray(mask_config[value['mask_selection']][
                                                                                  "boundary"]).convert('RGB'))
                                image = cv2.addWeighted(image.astype(np.uint8), 1, reconverted.astype(np.uint8), 1, 0)
                        region = np.array(image[np.ix_(range(int(y_min), int(y_max), 1),
                                               range(int(x_min), int(x_max), 1))]).astype(np.uint8)
                        aspect_ratio = image.shape[1] / image.shape[0]
                        # set height based on the pixel number
                        height = 0.02 * image.shape[1] if 0.02 * image.shape[1] < 30 else 30
                        width = height * aspect_ratio
                        # first value is the width, second is the height
                        fig = plt.figure(figsize=(width, height))
                        fig.tight_layout()
                        # ax = fig.add_subplot(111)
                        # plt.axes((.1, .4, .8, .5))
                        ax = fig.add_axes((0, .4, 1, 0.5))
                        ax.imshow(region, interpolation='nearest')
                        ax.set_title(value['title'], fontsize=(width + 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        x_dims = float(x_max) - float(x_min)
                        y_dims = float(y_max) - float(y_min)
                        patches = []
                        for channel in value['channels']:
                            label = aliases[channel] if channel in aliases.keys() else channel
                            if blend_dict is not None:
                                try:
                                    col_use = blend_dict[channel]['color']
                                except KeyError:
                                    col_use = 'white'
                            else:
                                col_use = 'white'
                            patches.append(mpatches.Patch(color=col_use, label=label))
                        body = str(value['body']).replace(r'\n', '\n')
                        description = "Description:\n" + body + "\n\n" + "" \
                                "Region dimensions: " + str(int(x_dims)) + "x" + str(int(y_dims))
                        text_offset = .3 if height < 25 else .2
                        fig.text(.15, text_offset, description, fontsize=width)
                        fig.legend(handles=patches, fontsize=width, title='Channel List', title_fontsize=(width + 5))
                        # ax.set_xlabel(description, fontsize=25)
                        # y_offset = 0.95
                        # plt.figtext(0.01, 1, "Channels", size=16)
                        # for channel in channel_list:
                        #     plt.figtext(0.01, y_offset, channel, size=14)
                        #     y_offset -= 0.05
                pdf.savefig()
            return file_output
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

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


def identify_column_matching_roi_to_quantification(data_selection, quantification_frame, dataset_options):
    """
    Parse the quantification sheet and current ROI name to identify the column name to use to match
    the current ROI to the quantification sheet. Options are either `description` or `sample`. Description is
    prioritized as the name of the ROI, and sample is the file name with a 1-indexed counter such as {file_name}_1
    """
    quantification_frame = pd.DataFrame(quantification_frame)
    exp, slide, acq = split_string_at_pattern(data_selection)
    if 'description' in quantification_frame.columns and acq in quantification_frame['description'].tolist():
        return acq, 'description'
    elif 'sample' in quantification_frame.columns:
        try:
            index = dataset_options.index(data_selection) + 1
            sample_name = f"{exp}_{index}"
            return sample_name, 'sample'
        except IndexError:
            return None, None
    else:
        return None, None

def generate_mask_with_cluster_annotations(mask_array: np.array, cluster_frame: pd.DataFrame, cluster_annotations: dict,
                                           cluster_col: str = "cluster", cell_id_col: str = "cell_id", retain_cells=True):
    """
    Generate a mask where cluster annotations are filled in with a specified colour, and non-annotated cells
    remain as greyscale values
    Returns a mask in RGB format
    """
    cluster_frame = pd.DataFrame(cluster_frame)
    empty = np.zeros((mask_array.shape[0], mask_array.shape[1], 3))
    for cell_type in cluster_frame[cluster_col].unique().tolist():
        cell_list = cluster_frame[(cluster_frame[cluster_col] == cell_type)][cell_id_col].tolist()
        annot_mask = np.where(np.isin(mask_array, cell_list), mask_array, 0)
        annot_mask = recolour_greyscale(annot_mask, cluster_annotations[cell_type])
        empty = empty + annot_mask
    # Find where the cells are annotated, and add back in the ones that are not
    if retain_cells:
        already_cells = np.array(Image.fromarray(empty.astype(np.uint8)).convert('L')) != 0
        mask_array[already_cells] = 0
        # px.imshow(Image.fromarray(mask_array).convert('RGB')).show()
        return (empty + np.array(Image.fromarray(mask_array).convert('RGB'))).clip(0, 255).astype(np.uint8)
    else:
        return empty.astype(np.uint8)
