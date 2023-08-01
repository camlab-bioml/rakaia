
from dash_extensions.enrich import Serverside
from ..utils.cell_level_utils import *
from sklearn.preprocessing import StandardScaler
import sys
from ..utils.pixel_level_utils import *
from dash.exceptions import PreventUpdate

def get_pixel(mask, i, j):
    if len(mask.shape) > 2:
        return mask[i][j][0]
    else:
        return mask[i][j]

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

def return_umap_dataframe_from_quantification_dict(quantification_dict):
    if quantification_dict is not None:
        # TODO: process quantification by removing cells outside of the percentile range for pixel intensity (
        #  column-wise, by channel)
        data_frame = pd.DataFrame(quantification_dict)
        umap_obj = None
        for elem in ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']:
            if elem in data_frame.columns:
                data_frame = data_frame.drop([elem], axis=1)
        # TODO: evaluate the umap import speed (slow) possibly due to numba compilation:
        # https://github.com/lmcinnes/umap/issues/631
        if 'umap' not in sys.modules:
            import umap
        try:
            umap_obj = umap.UMAP()
        except UnboundLocalError:
            import umap
            umap_obj = umap.UMAP()
        if umap_obj is not None:
            scaled = StandardScaler().fit_transform(data_frame)
            embedding = umap_obj.fit_transform(scaled)
            return Serverside(embedding), list(data_frame.columns)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

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
