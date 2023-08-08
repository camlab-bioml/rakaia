import pandas as pd
from dash.exceptions import PreventUpdate
import os
from tifffile import TiffFile
import numpy as np
from dash_extensions.enrich import Serverside
from PIL import Image
from ..utils.cell_level_utils import *

def validate_incoming_measurements_csv(measurements_csv, current_image=None, validate_with_image=True,
                                       required_columns=['cell_id', 'x', 'y', 'x_max', 'y_max', 'area']):
    """
    Validate an incoming measurements CSV against the current canvas, and ensure that it has the required
    information columns
    """
    if not all([column in measurements_csv.columns for column in required_columns]):
        return None
    # check the measurement CSV against an image to ensure that the dimensions match
    elif validate_with_image and current_image is not None:
        if float(current_image.shape[0]) != float(measurements_csv['x_max'].max()) or \
            float(current_image.shape[1]) != float(measurements_csv['y_max'].max()):
            return None
        else:
            return measurements_csv
    else:
        return measurements_csv

def filter_measurements_csv_by_channel_percentile(measurements, percentile=0.999,
                                                  dropped_columns=['cell_id', 'x', 'y', 'x_max',
                                                                   'y_max', 'area', 'sample'],
                                                  drop_cols=True):
    """
    Filter out the rows (cells) of a measurements csv (columns as channels, rows as cells) based on a pixel intensity
    threshold by percentile. Effectively removes any cells with "hot" pixels
    """
    if drop_cols:
        try:
            for col in dropped_columns:
                measurements = pd.DataFrame(measurements).drop(col, axis=1)
        except KeyError:
            pass

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


def parse_and_validate_measurements_csv(session_dict):
    """
    Validate the measurements CSV and return a clean version
    """
    if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
        quantification_worksheet = validate_incoming_measurements_csv(pd.read_csv(session_dict['uploads'][0]),
                                                                      validate_with_image=False)
        # TODO: establish where to use the percentile filtering on the measurements
        return filter_measurements_csv_by_channel_percentile(quantification_worksheet).to_dict(orient="records"), \
            list(pd.read_csv(session_dict['uploads'][0]).columns)
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
                cur_mask_dict[chosen_mask_name] = {"array": mask_import, "boundary": boundary_import}
        return Serverside(cur_mask_dict), list(cur_mask_dict.keys())
    else:
        raise PreventUpdate
