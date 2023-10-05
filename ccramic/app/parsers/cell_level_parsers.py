import dash
import pandas as pd
from dash.exceptions import PreventUpdate
import os
from tifffile import TiffFile
import numpy as np
from dash_extensions.enrich import Serverside
from PIL import Image
from ..utils.cell_level_utils import set_columns_to_drop
from ..utils.cell_level_utils import *
from pathlib import Path
import anndata

def drop_columns_from_measurements_csv(measurements_csv,
                                       cols_to_drop=set_columns_to_drop()):
    cols_to_drop = set_columns_to_drop(measurements_csv)
    try:
        for col in cols_to_drop:
            if col in measurements_csv.columns:
                measurements_csv = pd.DataFrame(measurements_csv).drop(col, axis=1)
        return measurements_csv
    except KeyError:
        return measurements_csv

def return_umap_dataframe_from_quantification_dict(quantification_dict, current_umap=None, drop_col=True):
    if quantification_dict is not None:
        data_frame = pd.DataFrame(quantification_dict)
        cols = list(data_frame.columns)
        if current_umap is None:
                # TODO: process quantification by removing cells outside of the percentile range for pixel intensity (
            #  column-wise, by channel)
            umap_obj = None
            if drop_col:
                data_frame = drop_columns_from_measurements_csv(data_frame)
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
                return Serverside(embedding), cols
            else:
                raise PreventUpdate
        else:
            return dash.no_update, cols
    else:
        raise PreventUpdate


def validate_incoming_measurements_csv(measurements_csv, current_image=None, validate_with_image=True,
                                       required_columns=set_mandatory_columns()):
    """
    Validate an incoming measurements CSV against the current canvas, and ensure that it has the required
    information columns
    """
    if not all([column in measurements_csv.columns for column in required_columns]):
        return None, None
    #TODO: find a different heuristic for validating the measurements CSV as it will contain multiple ROIs
    # # check the measurement CSV against an image to ensure that the dimensions match
    # elif validate_with_image and current_image is not None:
    #     if float(current_image.shape[0]) != float(measurements_csv['x_max'].max()) or \
    #         float(current_image.shape[1]) != float(measurements_csv['y_max'].max()):
    #         return measurements_csv, "Warning: the dimensions of the current ROI do not match the quantification sheet."
    #     else:
    #         return measurements_csv, None
    else:
        return measurements_csv, None

def filter_measurements_csv_by_channel_percentile(measurements, percentile=0.999,
                                                  drop_cols=False):
    """
    Filter out the rows (cells) of a measurements csv (columns as channels, rows as cells) based on a pixel intensity
    threshold by percentile. Effectively removes any cells with "hot" pixels
    """
    if drop_cols:
        measurements = drop_columns_from_measurements_csv(measurements)

    query = ""
    quantiles = measurements.quantile(q=percentile)
    channel_index = 0
    for index, value in quantiles.items():
        query = query + f"`{index}` < {value}"
        if channel_index < len(quantiles) - 1:
            query = query + " & "
        channel_index += 1

    filtered = measurements.query(query, engine='python')
    return pd.DataFrame(filtered)

def get_quantification_filepaths_from_drag_and_drop(status):
    filenames = [str(x) for x in status.uploaded_files]
    session_config = {'uploads': []}
    # IMP: ensure that the progress is up to 100% in the float before beginning to process
    if filenames and float(status.progress) == 1.0:
        for file in filenames:
            session_config['uploads'].append(file)
        return session_config
    else:
        raise PreventUpdate


def parse_and_validate_measurements_csv(session_dict, error_config=None, image_to_validate=None,
                                        use_percentile=False):
    """
    Validate the measurements CSV and return a clean version
    Use percentile filtering for removing hot pixel cells
    """
    if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
        if str(session_dict['uploads'][0]).endswith('.csv'):
            quantification_worksheet, warning = validate_incoming_measurements_csv(pd.read_csv(session_dict['uploads'][0]),
                                            current_image=image_to_validate, validate_with_image=True)
        elif str(session_dict['uploads'][0]).endswith('.h5ad'):
            quantification_worksheet, warning = validate_quantification_from_anndata(session_dict['uploads'][0])
        else:
            quantification_worksheet, warning = None, "Error: could not find a valid quantification sheet."
        # TODO: establish where to use the percentile filtering on the measurements
        measurements_return = filter_measurements_csv_by_channel_percentile(
            quantification_worksheet).to_dict(orient="records") if use_percentile else \
            quantification_worksheet.to_dict(orient="records")
        warning_return = dash.no_update
        if warning is not None:
            if error_config is None:
                error_config = {"error": None}
            error_config["error"] = warning
            warning_return = error_config
        return measurements_return, quantification_worksheet.columns, warning_return
    else:
        raise PreventUpdate


def parse_masks_from_filenames(status):
    filenames = [str(x) for x in status.uploaded_files]
    # IMP: ensure that the progress is up to 100% in the float before beginning to process
    if len(filenames) == 1:
        default_mask_name = os.path.splitext(os.path.basename(filenames[0]))[0]
        return {default_mask_name: filenames[0]}
    else:
        raise PreventUpdate

def read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict, derive_cell_boundary):
    if set_mask > 0 and None not in (mask_uploads, chosen_mask_name):
        cur_mask_dict = {} if cur_mask_dict is None else cur_mask_dict
        with TiffFile(str(mask_uploads[list(mask_uploads.keys())[0]])) as tif:
            for page in tif.pages:
                if derive_cell_boundary:
                    mask_import = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(page.asarray())).convert('RGB'))
                    boundary_import = None
                else:
                    mask_import = np.array(Image.fromarray(page.asarray()).convert('RGB'))
                    boundary_import = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(page.asarray())).convert('RGB'))
                cur_mask_dict[chosen_mask_name] = {"array": mask_import, "boundary": boundary_import,
                                                   "hover": page.asarray().reshape((page.asarray().shape[0],
                                                                                    page.asarray().shape[1], 1)),
                                                   "raw": page.asarray()}
        return Serverside(cur_mask_dict), list(cur_mask_dict.keys())
    else:
        raise PreventUpdate


def validate_quantification_from_anndata(anndata_obj, required_columns=set_mandatory_columns()):
    obj = anndata.read_h5ad(anndata_obj)
    frame = pd.DataFrame(obj.obs)
    if not all([column in frame.columns for column in required_columns]):
        return None, None
    else:
        return frame, None
