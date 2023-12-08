import dash
import pandas as pd
from dash.exceptions import PreventUpdate
import os
from tifffile import TiffFile
import numpy as np
from dash_extensions.enrich import Serverside
from PIL import Image
from ccramic.utils.cell_level_utils import (
    set_mandatory_columns,
    convert_mask_to_cell_boundary,
    set_columns_to_drop)
from ccramic.utils.pixel_level_utils import split_string_at_pattern
import anndata
import sys
from sklearn.preprocessing import StandardScaler

def drop_columns_from_measurements_csv(measurements_csv):
    cols_to_drop = set_columns_to_drop(measurements_csv)
    try:
        for col in cols_to_drop:
            if col in measurements_csv.columns:
                measurements_csv = pd.DataFrame(measurements_csv).drop(col, axis=1)
        return measurements_csv
    except KeyError:
        return measurements_csv

def return_umap_dataframe_from_quantification_dict(quantification_dict, current_umap=None, drop_col=True,
                                                   rerun=True):
    if quantification_dict is not None:
        data_frame = pd.DataFrame(quantification_dict)
        cols = list(data_frame.columns)
        if current_umap is None or rerun:
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


def validate_incoming_measurements_csv(measurements_csv, required_columns=set_mandatory_columns()):
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


def parse_and_validate_measurements_csv(session_dict, error_config=None,
                                        use_percentile=False):
    """
    Validate the measurements CSV and return a clean version
    Use percentile filtering for removing hot pixel cells
    """
    if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
        if str(session_dict['uploads'][0]).endswith('.csv'):
            quantification_worksheet, warning = validate_incoming_measurements_csv(pd.read_csv(session_dict['uploads'][0]))
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
    # TODO: establish multi mask import
    # if len(filenames) == 1:
    #     default_mask_name = os.path.splitext(os.path.basename(filenames[0]))[0]
    #     return {default_mask_name: filenames[0]}
    masks = {}
    for mask_file in filenames:
        default_mask_name = os.path.splitext(os.path.basename(mask_file))[0]
        masks[default_mask_name] = mask_file
    if len(masks) > 0:
        return masks
    else:
        raise PreventUpdate

def read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict, derive_cell_boundary):
    #TODO: establish parsing for single mask upload and bulk
    single_upload = len(mask_uploads) == 1 and set_mask > 0
    multi_upload = len(mask_uploads) > 1
    if single_upload or multi_upload:
        if 0 < len(mask_uploads) <= 1:
            single_mask_name = chosen_mask_name
        else:
            single_mask_name = None
        cur_mask_dict = {} if cur_mask_dict is None else cur_mask_dict
        for mask_name, mask_upload in mask_uploads.items():
            with TiffFile(str(mask_upload)) as tif:
                for page in tif.pages:
                    if derive_cell_boundary:
                        mask_import = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(page.asarray())).convert('RGB'))
                        boundary_import = None
                    else:
                        mask_import = np.array(Image.fromarray(page.asarray()).convert('RGB'))
                        boundary_import = np.array(Image.fromarray(
                            convert_mask_to_cell_boundary(page.asarray().astype(np.uint32))).convert('RGB'))
                    mask_name_use = single_mask_name if single_mask_name is not None else mask_name
                    cur_mask_dict[mask_name_use] = {"array": mask_import, "boundary": boundary_import,
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

def parse_cell_subtypes_from_restyledata(restyledata, quantification_frame, umap_col_annotation, existing_cats=None):
    """
    Parse the selected cell subtypes from the UMAP plot as selected by the legend
    if a subset is not found, return None
    """
    # Example 1: user selected only the third legend item to view
    # [{'visible': ['legendonly', 'legendonly', True, 'legendonly', 'legendonly', 'legendonly', 'legendonly']}, [0, 1, 2, 3, 4, 5, 6]]
    # Example 2: user selects all but the the second item to view
    # [{'visible': ['legendonly']}, [2]]
    # print(restyle_data)
    if None not in (restyledata, quantification_frame) and 'visible' in restyledata[0] and \
        umap_col_annotation is not None and umap_col_annotation in list(pd.DataFrame(quantification_frame).columns):
        # get the total number of possible sub annotations and figure out which ones were selected
        quant_frame = pd.DataFrame(quantification_frame)
        tot_subtypes = list(quant_frame[umap_col_annotation].unique())
        subtypes_keep = []
        # Case 1: if only one sub type is selected
        if len(restyledata[0]['visible']) == len(tot_subtypes):
            indices_use = []
            for selection in range(len(restyledata[0]['visible'])):
                if restyledata[0]['visible'][selection] != 'legendonly':
                    subtypes_keep.append(tot_subtypes[selection])
                    indices_use.append(selection)
            return subtypes_keep, indices_use
        # TODO: different options for single select (include or not)
        # Case 2: if the user has already added or excluded one sub type

        # Case 2.1: when user wants to remove current index plus other ones that have already been removed
        # [{'visible': ['legendonly']}, [3]]

        # Case 2.2: when user wants to add current index plus others that have already been added
        # [{'visible': [True]}, [3]]
        elif len(restyledata[0]['visible']) == 1 and len(restyledata[1]) == 1:
            # case 2.1: When the current and previous indices are to be ignored
            # [{'visible': ['legendonly']}, [3]]
            if restyledata[0]['visible'][0] == 'legendonly':
                # existing indices will be ones to keep
                indices_keep = existing_cats.copy() if existing_cats is not None else \
                    [ind for ind in range(0, len(tot_subtypes))]
                if restyledata[1][0] in indices_keep:
                    indices_keep.remove(restyledata[1][0])
                for selection in range(len(tot_subtypes)):
                    if selection in indices_keep:
                        subtypes_keep.append(tot_subtypes[selection])
                return subtypes_keep, indices_keep
            # Case 2.2: when the current and previous indices are to be kept
            # [{'visible': [True]}, [3]]
            elif restyledata[0]['visible'][0]:
                indices_keep = existing_cats.copy() if existing_cats is not None else []
                for elem in restyledata[1]:
                    if elem not in indices_keep:
                        indices_keep.append(elem)
                for selection in range(len(tot_subtypes)):
                    if selection in indices_keep:
                        subtypes_keep.append(tot_subtypes[selection])
                return subtypes_keep, indices_keep
    else:
        return None, None


def parse_roi_query_indices_from_quantification_subset(quantification_dict, subset_frame, umap_col_selection=None):
    """
    Parse the ROIs included in a view of a subset quantification sheet
    """
    # get the merged frames to pull the sample names in the subset
    full_frame = pd.DataFrame(quantification_dict)
    merged = subset_frame.merge(full_frame, how="inner", on=subset_frame.columns.tolist())
    # get the roi names from either the description or the sample name
    if 'description' in list(merged.columns):
        indices_query = {'names': list(merged['description'].value_counts().to_dict().keys())}
    else:
        try:
            roi_counts = merged['sample'].value_counts().to_dict()
            indices_query = {'indices': [int(i.split("_")[-1]) - 1 for i in list(roi_counts.keys())]}
        except ValueError:
            # may occur if the split doesn't give integers i.e. if there are other underscores in the name
            indices_query = None

    freq_counts = merged[umap_col_selection].value_counts().to_dict() if umap_col_selection is \
                    not None else None

    return indices_query, freq_counts


def match_mask_name_with_roi(data_selection, mask_options, roi_options):
    """
    Attempt to match a mask name to the currently selected ROI.
    Heuristics order:
    1. If the data selection experiment name is in the list of mask options, return it
    2. If the data selection ROI name is in the list of mask options, return it
    3. If any of the mask names have indices in them, return the ROI name at that index
    4. If None of those exist, return None
    """
    mask_return = None
    if mask_options is not None and data_selection in mask_options:
        mask_return = data_selection
    else:
        # first, check to see if the pattern matches based on the pipeline mask name output
        if mask_options is not None and roi_options is not None:
            data_index = roi_options.index(data_selection)
            for mask in mask_options:
                try:
                    # mask naming fro the pipeline follows {mcd_name}_s0_a2_ac_IA_mask.tiff
                    # where s0 is the slide index (0-indexed) and a2 is the acquisition index (1-indexed)
                    split_1 = mask.split("_ac_IA_mask")[0]
                    index = int(split_1.split("_")[-1].replace("a", "")) - 1
                    if index == data_index:
                        mask_return = mask
                except (TypeError, IndexError, ValueError):
                    pass
        if mask_return is None and "+++" in data_selection:
            exp, slide, acq = split_string_at_pattern(data_selection)
            if mask_options is not None and exp in mask_options:
                mask_return = exp
            elif mask_options is not None and acq in mask_options:
                mask_return = acq
            # begin looking for partial matches if not direct match between experiment/ROI name and mask
            elif mask_options is not None:
                for mask_name in mask_options:
                    # check if any overlap with the mask options and the current data selection
                    # IMP: assumes that the mask name contains all of the experiment or ROI name somewhere in the label
                    if exp in mask_name or acq in mask_name:
                        mask_return = mask_name
    return mask_return


def match_mask_name_to_quantification_sheet_roi(mask_selection, cell_id_list, sample_col_id="sample"):
    """
    Match a mask name to a sample ID in the quantification sheet, either in the `description` or `sample` table
    Example: query_s0_a2_ac_IA_mask will match to query_2 in the quantification sheet
    """
    sam_id = None
    if mask_selection in cell_id_list:
        sam_id = mask_selection
    else:
        # if this pattern exists, try to match to the sample name by index
        # otherwise, try matching directly by name
        if "_ac_IA_mask" in mask_selection:
            try:
                split_1 = mask_selection.split("_ac_IA_mask")[0]
                # IMP: do not subtract 1 here as both the quantification sheet and mask name are 1-indexed
                index = int(split_1.split("_")[-1].replace("a", ""))
                for sample in sorted(cell_id_list):
                    if sample == mask_selection:
                        sam_id = sample
                    elif sample_col_id == "sample":
                        split = sample.split("_")
                        try:
                            if int(split[-1]) == int(index):
                                sam_id = sample
                        except ValueError:
                            pass
                return sam_id
            except (KeyError, TypeError):
                pass
        # otherwise, try to match the mask name exactly
        else:
            #TODO: implement logic for finding partial mask matches to quantificcation sheets
            pass
    return sam_id

def validate_imported_csv_annotations(annotations_csv):
    """
    Validate that the imported annotations CSV has the correct format and columns
    """
    frame = pd.DataFrame(annotations_csv)
    return "x" in list(frame.columns) and "y" in list(frame.columns)


def validate_coordinate_set_for_image(x=None, y=None, image=None):
    """
    Validate that a pair of xy coordinates can fit inside an image's dimensions
    """
    if None not in (x, y) and image is not None:
        return int(x) <= image.shape[1] and int(y) <= image.shape[0]
    else:
        return False
