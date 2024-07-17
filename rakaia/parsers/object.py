import pandas as pd
from dash.exceptions import PreventUpdate
import os
from tifffile import TiffFile
import numpy as np
from PIL import Image
import anndata
import sys
from sklearn.preprocessing import StandardScaler
import scanpy as sc
from typing import Union
import re
import dash
from rakaia.utils.object import (
    set_mandatory_columns,
    convert_mask_to_object_boundary,
    set_columns_to_drop,
    QuantificationColumns,
    QuantificationFormatError,
    match_steinbock_mask_name_to_mcd_roi,
    is_steinbock_intensity_anndata)
from rakaia.utils.pixel import split_string_at_pattern
from rakaia.io.session import SessionServerside

def drop_columns_from_measurements_csv(measurements_csv):
    cols_to_drop = set_columns_to_drop(measurements_csv)
    for col in cols_to_drop:
        if col in measurements_csv.columns:
            measurements_csv = pd.DataFrame(measurements_csv).drop(col, axis=1)
    return measurements_csv

def return_umap_dataframe_from_quantification_dict(quantification_dict, current_umap=None, drop_col=True,
                                                   rerun=True, unique_key_serverside=True,
                                                   cols_include: list=None):
    """
    Generate a UMAP coordinate frame from a data frame of channel expression
    `cols_include`: Pass an optional list of channels to generate the coordinates from
    """
    if quantification_dict is not None:
        data_frame = pd.DataFrame(quantification_dict)
        if cols_include:
            data_frame = data_frame[data_frame.columns.intersection(cols_include)]
        if not data_frame.empty and (current_umap is None or rerun):
            if drop_col:
                data_frame = drop_columns_from_measurements_csv(data_frame)
            # the umap import speed (slow) possibly due to numba compilation, so only load when the function is called
            # https://github.com/lmcinnes/umap/issues/631
            if 'umap' not in sys.modules:
                import umap
            try:
                umap_obj = umap.UMAP()
            except UnboundLocalError:
                import umap
                umap_obj = umap.UMAP()
            if umap_obj:
                scaled = StandardScaler().fit_transform(data_frame)
                embedding = umap_obj.fit_transform(scaled)
                return SessionServerside(embedding, key="umap-embedding",
                                         use_unique_key=unique_key_serverside)
            return dash.no_update
        return dash.no_update
    return dash.no_update


def validate_incoming_measurements_csv(measurements_csv, required_columns=set_mandatory_columns()):
    """
    Validate an incoming measurements CSV against the current canvas, and ensure that it has the required
    information columns
    """
    if not any([column in measurements_csv.columns for column in QuantificationColumns().identifiers]):
        raise QuantificationFormatError(
            "The imported quantification results are missing at least one of the following:\n"
            "`sample` or `description`, which should immediately follow the channel "
            "columns in the CSV, and should link ROI names/masks to quantification results.")
    if not all([column in measurements_csv.columns for column in required_columns]):
        return None, None
    else:
        return measurements_csv, None

def filter_measurements_csv_by_channel_percentile(measurements, percentile=0.999, drop_cols=False):
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
    raise PreventUpdate


def parse_and_validate_measurements_csv(session_dict, error_config=None, use_percentile=False):
    """
    Validate the measurements CSV and return a clean version
    Use percentile filtering for removing hot pixel cells
    """
    if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
        if str(session_dict['uploads'][0]).endswith('.csv'):
            quantification_worksheet, warning = validate_incoming_measurements_csv(pd.read_csv(session_dict['uploads'][0]))
        elif str(session_dict['uploads'][0]).endswith('.h5ad'):
            quantification_worksheet, warning = validate_quantification_from_anndata(
                parse_quantification_sheet_from_h5ad(session_dict['uploads'][0]))
        else:
            quantification_worksheet, warning = None, "Error: could not find a valid quantification sheet."
        measurements_return = filter_measurements_csv_by_channel_percentile(
            quantification_worksheet).to_dict(orient="records") if use_percentile else \
            quantification_worksheet.to_dict(orient="records") if quantification_worksheet is not None else None
        cols_return = quantification_worksheet.columns if quantification_worksheet is not None else None
        warning_return = dash.no_update
        if warning is not None:
            if error_config is None:
                error_config = {"error": None}
            error_config["error"] = warning
            warning_return = error_config
        return measurements_return, cols_return, warning_return
    raise PreventUpdate


def parse_masks_from_filenames(status):
    filenames = [str(x) for x in status.uploaded_files]
    masks = {}
    if status.progress == 1.0:
        for mask_file in filenames:
            default_mask_name = os.path.splitext(os.path.basename(mask_file))[0]
            masks[default_mask_name] = mask_file
        if len(masks) > 0:
            return masks
    raise PreventUpdate

def read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name,
                                     set_mask, cur_mask_dict, derive_cell_boundary=False, unique_key_serverside=True):
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
                        convert_mask_to_object_boundary(page.asarray())).convert('RGB'))
                        boundary_import = None
                    else:
                        mask_import = np.array(Image.fromarray(page.asarray()).convert('RGB'))
                        boundary_import = np.array(Image.fromarray(
                            convert_mask_to_object_boundary(page.asarray().astype(np.uint32))).convert('RGB'))
                    mask_name_use = single_mask_name if single_mask_name is not None else mask_name
                    cur_mask_dict[mask_name_use] = {"array": mask_import, "boundary": boundary_import,
                                                   "hover": page.asarray().reshape((page.asarray().shape[0],
                                                                                    page.asarray().shape[1], 1)),
                                                   "raw": page.asarray()}
        return SessionServerside(cur_mask_dict, key="mask-dict",
                                 use_unique_key=unique_key_serverside), list(cur_mask_dict.keys())
    raise PreventUpdate


def validate_quantification_from_anndata(anndata_obj, required_columns=set_mandatory_columns()):
    if isinstance(anndata_obj, str):
        obj = anndata.read_h5ad(anndata_obj)
        frame = pd.DataFrame(obj.obs)
    else:
        frame = anndata_obj
    if not all([column in frame.columns for column in required_columns]):
        return None, None
    else:
        return frame, None

class RestyleDataParser:
    """
    Parse the selected categorical subtypes from the UMAP plot as selected by the legend.
    Categorical selections from the interactive legend are found in the restyledata attribute of the
    `go.Figure` object, stored as index lists under the 'visible' hash key
    If a subset is not found, return None for both elements of the tuple
    Returns a tuple of two lists: a list of subtypes in string form to retain from the graph,
    and a list of the indices of those subtypes relative to the list of subtypes for a specific category

    :param resyledata: Dictionary from the UMAP `go.Figure` object specifying legend category interaction
    :param quantification_frame: dictionary of `pd.DataFrame` for quantified mask objects
    :param umap_col_annotation: The current UMAP overlay category (found in `quantification_frame`).
    :param existing_categories: List of any previously selected variables in the current UMAP overlay category
    :return: None
    """
    def __init__(self, restyledata: Union[dict, list], quantification_frame: Union[dict, pd.DataFrame],
                 umap_col_annotation: str, existing_categories: Union[dict, list]=None):
        self.restyledata = restyledata
        self.quantification_frame = quantification_frame
        self.umap_col_annotation = umap_col_annotation
        self.existing_categories = existing_categories
        self._subtypes_return = None
        self._indices_keep = None

        if None not in (self.restyledata, self.quantification_frame) and 'visible' in self.restyledata[0] and \
               self.umap_col_annotation is not None and self.umap_col_annotation in list(
            pd.DataFrame(self.quantification_frame).columns) and \
                self.restyledata not in [[{'visible': ['legendonly']}, [0]]]:
            quant_frame = pd.DataFrame(self.quantification_frame)
            tot_subtypes = list(quant_frame[self.umap_col_annotation].unique())
            self._subtypes_return = []
            if self.single_category_selected(tot_subtypes):
                self._indices_keep = []
                for selection in range(len(self.restyledata[0]['visible'])):
                    if self.restyledata[0]['visible'][selection] != 'legendonly':
                        self._subtypes_return.append(tot_subtypes[selection])
                        self._indices_keep.append(selection)
            elif self.subtypes_already_selected():
                if self.ignore_selected_index():
                    self.generate_indices_from_ignore(self._subtypes_return, tot_subtypes)
                elif self.keep_current_index():
                    self.generate_indices_from_keep(self._subtypes_return, tot_subtypes)

    def single_category_selected(self, tot_subtypes):
        """
        :return: If the restyledata returns a single category selection
        """
        return len(self.restyledata[0]['visible']) == len(tot_subtypes)

    def subtypes_already_selected(self):
        """
        :return: If the restyledata indicates that categories have already been selected
        """
        return len(self.restyledata[0]['visible']) == 1 and len(self.restyledata[1]) == 1

    def ignore_selected_index(self):
        """
        :return: If the retyledata ignores the currently selected index
        """
        return self.restyledata[0]['visible'][0] == 'legendonly'

    def generate_indices_from_ignore(self, subtypes_keep, tot_subtypes):
        """
        Specify a list of subtypes and their ordinal indices to ignore
        :return: None
        """
        indices_keep = self.existing_categories.copy() if self.existing_categories is not None else \
            [ind for ind in range(0, len(tot_subtypes))]
        if self.restyledata[1][0] in indices_keep:
            indices_keep.remove(self.restyledata[1][0])
        for selection in range(len(tot_subtypes)):
            if selection in indices_keep:
                subtypes_keep.append(tot_subtypes[selection])
        self._subtypes_return, self._indices_keep = subtypes_keep, indices_keep

    def keep_current_index(self):
        """
        :return: If the currently selected category index should be kept.
        """
        return bool(self.restyledata[0]['visible'][0])

    def generate_indices_from_keep(self, subtypes_keep, tot_subtypes):
        """
        Specify a list of subtypes and their ordinal indices to keep.
        return: None
        """
        indices_keep = self.existing_categories.copy() if self.existing_categories is not None else []
        for elem in self.restyledata[1]:
            if elem not in indices_keep:
                indices_keep.append(elem)
        for selection in range(len(tot_subtypes)):
            if selection in indices_keep:
                subtypes_keep.append(tot_subtypes[selection])
        self._subtypes_return, self._indices_keep = subtypes_keep, indices_keep

    def get_callback_structures(self):
        """
        Get a list of subtypes to keep for analysis and their ordinal indices
        :return: tuple: List of subtypes to keep, list of corresponding indices to keep
        """
        return self._subtypes_return if self._subtypes_return else None, \
            self._indices_keep if self._indices_keep else None

def parse_roi_query_indices_from_quantification_subset(quantification_dict, subset_frame, umap_col_selection=None):
    """
    Parse the ROIs included in a view of a subset quantification sheet
    Returns a tuple: a dictionary of subtype keys, and a dictionary of frequency counts for those keys
    from the quantification results
    """
    # merged = pd.concat([subset_frame, pd.DataFrame(quantification_dict)], axis=1,
    #                    join="inner").reset_index(drop=True)
    merged = pd.DataFrame(quantification_dict).iloc[list(subset_frame.index.values)]
    if 'description' in list(merged.columns):
        indices_query = {'names': list(merged['description'].value_counts().to_dict().keys())}
    else:
        try:
            roi_counts = merged['sample'].value_counts().to_dict()
            indices_query = {'indices': [int(i.split("_")[-1]) - 1 for i in list(roi_counts.keys())]}
        # may occur if the split doesn't give integers i.e. if there are other underscores in the name
        except ValueError: indices_query = None
    freq_counts = merged[umap_col_selection].value_counts().to_dict() if umap_col_selection is \
                    not None else None
    return indices_query, freq_counts


def match_mask_name_with_roi(data_selection: str, mask_options: list, roi_options: list, delimiter: str="+++",
                             return_as_dash: bool=False):
    """
    Attempt to match a mask name to the currently selected ROI.
    Heuristics order:
    1. If the data selection experiment name is in the list of mask options, return it
    2. If the data selection ROI name is in the list of mask options, return it
    3. If any of the mask names have indices in them, return the ROI name at that index
    4. If None of those exist, return None
    mask options : list of masks in the current session
    roi options: list of dropdown ROI options in the current session
    """
    if mask_options is not None and data_selection in mask_options:
        mask_return = data_selection
    else:
        # first, check to see if the pattern matches based on the pipeline mask name output
        mask_return = match_mask_old_pipeline_syntax(data_selection, mask_options, roi_options)
        if not mask_return and delimiter in data_selection:
            exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
            mask_return = match_mask_to_roi_name_components(data_selection, mask_options, delimiter)
            # first loop: look for a match using the steinbock naming conventions
            if mask_options is not None and not mask_return:
                for mask_name in mask_options:
                    # check if any overlap with the mask options and the current data selection
                    # IMP: assumes that the mask name contains all of the experiment or ROI name somewhere in the label
                    if match_steinbock_mask_name_to_mcd_roi(mask_name, acq) and exp in mask_name:
                        mask_return = mask_name
                        break
            # second loop: begin looking for partial matches if not direct match between experiment/ROI name and mask
            if mask_options is not None and not mask_return:
                for mask_name in mask_options:
                    if exp in mask_name or acq in mask_name:
                        mask_return = mask_name
                        break
    if not mask_return:
        mask_return = dash.no_update if (return_as_dash and not mask_options) else None
    return mask_return

def match_mask_old_pipeline_syntax(data_selection: str, mask_options: Union[list, None],
                                   roi_options: Union[list, None]):
    # first, check to see if the pattern matches based on the pipeline mask name output
    mask_return = None
    if mask_options is not None and roi_options is not None:
        data_index = roi_options.index(data_selection)
        for mask in mask_options:
            try:
                if "_ac_IA_mask" in mask:
                    # mask naming from the pipeline follows {mcd_name}_s0_a2_ac_IA_mask.tiff
                    # where s0 is the slide index (0-indexed) and a2 is the acquisition index (1-indexed)
                    split_1 = mask.split("_ac_IA_mask")[0]
                    index = int(split_1.split("_")[-1].replace("a", "")) - 1
                    if index == data_index:
                        mask_return = mask
            except (TypeError, IndexError, ValueError):
                pass
    return mask_return

def match_mask_to_roi_name_components(data_selection: str, mask_options: Union[list, None],
                                      delimiter: str="+++"):
    mask_return = None
    exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
    if mask_options is not None and exp in mask_options:
        mask_return = exp
    elif mask_options is not None and acq in mask_options:
        mask_return = acq
    return mask_return


def match_mask_name_to_quantification_sheet_roi(mask_selection: str, cell_id_list: Union[list, None],
                                                sample_col_id="sample"):
    """
    Match a mask name to a sample ID in the quantification sheet, either in the `description` or `sample` table
    Example: query_s0_a2_ac_IA_mask will match to query_2 in the quantification sheet
    mask_selection: string representation of the current mask selection
    cell_id_list: list of ROIs that correspond to a subset of mask object ids in the subset. Could include 0 or more
    ROI string representations
    """
    sam_id = None
    if cell_id_list is not None and mask_selection in cell_id_list:
        sam_id = mask_selection
    else:
        # also look for partial match of the cell id list to the mark name
        if cell_id_list:
            for roi_id in cell_id_list:
                # if exact match
                if roi_id and mask_selection and roi_id in mask_selection:
                    sam_id = roi_id
                if not sam_id:
                    sam_id = match_steinbock_mask_name_to_mcd_roi(mask_selection, roi_id, False)
        # if this pattern exists, try to match to the sample name by index
        # otherwise, try matching directly by name
        if not sam_id:
            sam_id = match_quantification_identifier_old_pipeline_syntax(mask_selection, cell_id_list, sample_col_id)
    return sam_id

def match_quantification_identifier_old_pipeline_syntax(mask_selection: str, cell_id_list: Union[list, None],
                                                        sample_col_id: str="sample"):
    sam_id = None
    if mask_selection and "_ac_IA_mask" in mask_selection and cell_id_list:
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
    return sam_id

def validate_imported_csv_annotations(annotations_csv):
    """
    Validate that the imported annotations CSV has the correct format and columns
    """
    frame = pd.DataFrame(annotations_csv)
    return "x" in list(frame.columns) and "y" in list(frame.columns)


def validate_coordinate_set_for_image(x_coord=None, y_coord=None, image=None):
    """
    Validate that a pair of xy coordinates can fit inside an image's dimensions
    """
    if None not in (x_coord, y_coord) and image is not None:
        return int(x_coord) <= image.shape[1] and int(y_coord) <= image.shape[0]
    return False

def parse_quantification_sheet_from_h5ad(h5ad_file):
    """
    Parse the quantification results from an h5ad files. Assumes the following format:
    - Channel expression is held as an array in h5ad_file.X
    - Channel names are held in h5ad_file.var_names
    - Additional metadata variables are held in h5ad_file.obs
    """
    quantification_frame = sc.read_h5ad(h5ad_file)
    expression = pd.DataFrame(quantification_frame.X,
                columns=list(quantification_frame.var_names)).reset_index(
                drop=True)
    if is_steinbock_intensity_anndata(quantification_frame):
        # create a sample column that uses indices to match the steinbock masks
        edited = pd.DataFrame({"description": [f"{acq}_{position}" for acq, position in \
                                               zip(quantification_frame.obs['image_acquisition_description'],
        [int(re.search(r'\d+$', elem.split('.tiff')[0]).group()) for elem in quantification_frame.obs['Image']])],
                # parse the int cell id from the string index
                "cell_id": [int(re.search(r'\d+', elem).group()) for elem in quantification_frame.obs.index],
                "sample": [elem.split('.tiff')[0] for elem in quantification_frame.obs['Image']]},
                index=quantification_frame.obs.index)
        edited["cell_id"] = pd.to_numeric(edited["cell_id"])
        quantification_frame = pd.concat([edited, quantification_frame.obs], axis=1, ignore_index=False)
        return expression.join(quantification_frame.reset_index(drop=True))
    # return the merged version of the data frames to mimic the pipeline
    return expression.join(quantification_frame.obs.reset_index(drop=True))

def object_id_list_from_gating(gating_dict: dict, gating_selection: list,
                               quantification_frame: Union[dict, pd.DataFrame]=None,
                               mask_identifier: str=None, quantification_sample_col: str='sample',
                               quantification_object_col:str='cell_id', intersection=False, normalize=True):
    """
    Produce a list of object ids (i.e. cells from segmentation) from a mask that is gated on
    one or more quantifiable traits. Cell ids are used to subset the mask to show only gated cells
    gating dict has the form {channel: [lower_bound, upper_bound]} where the bounds indicate which cells
    should be retained with expression values in the channel range
    Can either use the intersection or union of gating on multiple traits
    """
    # set the possible mask names from the quantification sheet from either the description or the sample name
    designation_column = 'sample'
    try:
        if 'description' in quantification_frame.columns:
            id_list = quantification_frame['description'].to_list()
            designation_column = 'description'
        else:
            id_list = quantification_frame['sample'].to_list()
    except KeyError:
        id_list = None

    to_add = quantification_frame[quantification_frame.columns.intersection(
        [designation_column, quantification_object_col])]

    # set the query representation of intersection or union in query
    combo = "& " if intersection else "| "
    query = ""
    # build a query string for each of the elements to gate on
    gating_index = 0
    for gating_elem in gating_selection:
        if gating_elem not in to_add.columns and gating_elem in list(quantification_frame.columns):
            # do not add the string combo at the end
            combo = combo if gating_index < (len(gating_selection) - 1) else ""
            query = query + f'(`{gating_elem}` >= {gating_dict[gating_elem]["lower_bound"]} &' \
                        f'`{gating_elem}` <= {gating_dict[gating_elem]["upper_bound"]}) {combo}'
            gating_index += 1
    # set the mandatory columns that need to be appended for the search: including the ROI descriptor and
    # column to identify the cell ID
    mask_quant_match = None
    if None not in (mask_identifier, id_list):
        mask_quant_match = match_mask_name_to_quantification_sheet_roi(mask_identifier,
                                                id_list, quantification_sample_col)
    if mask_quant_match is not None and query:
        frame = quantification_frame
        if normalize:
            frame = quantification_frame[quantification_frame.columns.intersection(gating_selection)]
            frame = ((frame - frame.min()) / (frame.max() - frame.min()))
        frame = frame.reset_index(drop=True).join(to_add)
        query = frame.query(query)
        # pull the cell ids from the subset of the quantification frame from the query where the
        # mask matches the one provided
        cell_ids = [int(i) for i in query[query[designation_column] ==
                                          mask_quant_match][quantification_object_col].tolist()]
    else:
        cell_ids = []
    return cell_ids

def cluster_annotation_frame_import(cur_cluster_dict: dict=None, roi_selection: str=None, cluster_frame:
                                    Union[pd.DataFrame, dict]=None):
    cur_cluster_dict = {} if cur_cluster_dict is None else cur_cluster_dict
    cluster_frame = pd.DataFrame(cluster_frame)
    # for now, use set column names, but expand in the future
    if all([elem in list(cluster_frame.columns) for elem in ['cell_id', 'cluster']]):
        cur_cluster_dict[roi_selection] = cluster_frame
    return cur_cluster_dict
