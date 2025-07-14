"""Module containing functions and classes for object clustering within segmentation masks
"""

from typing import Union
import dash
import pandas as pd
from dash import html
from rakaia.utils.pixel import glasbey_palette
from rakaia.utils.object import ROIQuantificationMatch

class ClusterIdentifiers:
    """
    Set the set of mask object id column identifiers permitted for cluster projection
    """
    id_cols = ['cell_id', 'object_id']
    # define the column identifiers that can be used for multi ROI uploads
    multi_roi_cols = ['roi', 'description']

class QuantificationClusterMerge:
    """
    Transfer a category from quantification results to the cluster frame for a specific ROI.
    Requires that the quantification results have a sample identifying column that matches to the current ROI

    :param quantification_frame: Dictionary or `pd.Dataframe` of quantified object results across one or more ROIs
    :param roi_selection: String representation of the currently loaded ROI
    :param cat_to_transfer: List or string of quantification columns to add to cluster projection
    :param delimiter: string splitter to split the string ROI selection
    :param dataset_options: List of session ROIs. Used by old pipeline logic to match mask names by dataset index
    :return: None
    """
    def __init__(self, quantification_frame: Union[dict, pd.DataFrame], roi_selection: str,
                 cat_to_transfer: Union[str, list],
                 current_cluster_frame: Union[dict, pd.DataFrame]=None,
                 delimiter: str="+++", current_mask: str=None,
                 dataset_options: Union[list, None]=None) -> None:
        self.quantification_frame = pd.DataFrame(quantification_frame)
        self.roi_selection = roi_selection
        self.cat_to_transfer = cat_to_transfer if isinstance(cat_to_transfer, list) else [cat_to_transfer]
        self.delimiter = delimiter
        self.current_mask = current_mask
        self.dataset_options = dataset_options
        self._cluster_frame = current_cluster_frame if current_cluster_frame else {}
        # get the roi name match to the current ROI, and what column in the quant frame is used to link
        self.roi_match, self.quant_frame_identifier = ROIQuantificationMatch(self.roi_selection,
                                                        self.quantification_frame, self.dataset_options,
                                                        self.delimiter, self.current_mask).get_matches()
        # figure out which column is the quant results holds the object ids
        self.quant_object_identifier = get_cluster_proj_id_column(self.quantification_frame)
        self.quantification_frame[self.quant_object_identifier] = \
            self.quantification_frame[self.quant_object_identifier].astype(int)
        if self.cat_to_transfer and not self.quantification_frame.empty and all(elem in
                list(self.quantification_frame.columns) for elem in self.cat_to_transfer) and \
                self.roi_match and self.quant_frame_identifier:
            # make the subset of the quant frame for the current ROI
            subset = self.quantification_frame[self.quantification_frame[self.quant_frame_identifier] ==
                                               self.roi_match][[self.quant_object_identifier] + self.cat_to_transfer]
            self.set_new_cluster_frame(subset)

    def set_new_cluster_frame(self, new_clust: Union[dict, pd.DataFrame]):
        """
        Set the new cluster frame based on the presence of an existing frame

        :param new_clust: Cluster frame to replace or merge with existing results
        :return: None
        """
        # if no cluster frame, make a new one
        if not self._cluster_frame or self.roi_selection not in self._cluster_frame:
            self._cluster_frame[self.roi_selection] = new_clust
        # if cluster frame exists, merge
        else:
            clust_frame_identifier = get_cluster_proj_id_column(self._cluster_frame[self.roi_selection])
            cur_clust = self._cluster_frame[self.roi_selection]
            cur_clust[clust_frame_identifier] = cur_clust[clust_frame_identifier].astype(int)
            clust_return = pd.merge(cur_clust, new_clust, left_on=clust_frame_identifier,
                                    right_on=self.quant_object_identifier,
                                    how='inner')
            self._cluster_frame[self.roi_selection] = clust_return

    def get_cluster_frame(self) -> Union[dict, list]:
        """
        :return: Dictionary with the updated cluster frame for the current ROI as a dataframe.
        """
        return self._cluster_frame

def split_cluster_frame_upload_multi_roi(cur_cluster_dict: Union[dict,None]=None,
                                    cluster_frame: Union[pd.DataFrame, dict]=None,
                                    dataset_options: Union[list, None]=None,
                                    delimiter: str="+++"):
    # iterate the ROIs in the session to try to find a match
    dataset_options = dataset_options if dataset_options and isinstance(dataset_options, list) else []
    for roi in dataset_options:
        match, identifier = ROIQuantificationMatch(roi, cluster_frame,
                            dataset_options, delimiter, None,
                            ClusterIdentifiers.multi_roi_cols).get_matches()
        if match and identifier:
            cur_cluster_dict[roi] = cluster_frame[cluster_frame[identifier] == match]
    return cur_cluster_dict


def cluster_annotation_frame_import(cur_cluster_dict: Union[dict,None]=None, roi_selection: str=None,
                                    cluster_frame: Union[pd.DataFrame, dict]=None,
                                    dataset_options: Union[list, None]=None,
                                    delimiter: str="+++") -> dict:
    """
    Parse the column headers for an imported cluster annotation data frame and verify the presence of at least
    one column identifier
    """
    cur_cluster_dict = {} if not cur_cluster_dict else cur_cluster_dict
    cluster_frame = pd.DataFrame(cluster_frame)

    # Case 1: if description column there, means multi ROI. try to match up for every ROI
    if any(elem in list(cluster_frame.columns) for elem in ClusterIdentifiers.multi_roi_cols):
        return split_cluster_frame_upload_multi_roi(cur_cluster_dict, cluster_frame, dataset_options, delimiter)

    # Case 2: single ROI, map to current ROI only
    if any(elem in list(cluster_frame.columns) for elem in ClusterIdentifiers.id_cols) and roi_selection:
        cur_cluster_dict[roi_selection] = cluster_frame
    return cur_cluster_dict if len(cur_cluster_dict) > 0 else None

def subset_cluster_frame(cluster_data: dict, roi_selection: str, clust_variable: str,
                         cluster_cats: Union[list, None]=None,
                         gating_object_list: Union[list, None]=None) -> Union[pd.DataFrame, dict]:
    """
    Subset a cluster frame based on subset of cluster projection options in a specific column, or
    an object gating list. Used for generating a distribution table of objects by annotation in the current image
    """
    if cluster_data and roi_selection and clust_variable and cluster_cats and \
            roi_selection in cluster_data:
        cluster_cats = [str(i) for i in cluster_cats]
        cluster_data = pd.DataFrame(cluster_data[roi_selection])
        cluster_data[clust_variable] = cluster_data[clust_variable].apply(str)
        cluster_data = cluster_data[cluster_data[clust_variable].isin(list(cluster_cats))]
        if gating_object_list:
            object_column = get_cluster_proj_id_column(cluster_data)
            cluster_data = cluster_data[cluster_data[object_column].isin(gating_object_list)]
    return cluster_data

def assign_colours_to_cluster_annotations(cluster_frame_dict: dict=None, cur_cluster_dict: dict=None,
                                          roi_selection: str=None) -> Union[dict, None]:
    """
    Generate a dictionary of random colours to assign to the clusters for a specific ROI
    cluster frame dict contains the cluster assignments by ROI
    cur_cluster_dict contains current assignments from previous uploads or previous ROIs
    """
    try:
        cluster_assignments = {roi_selection: {}} if not cur_cluster_dict else cur_cluster_dict
        if roi_selection not in cluster_assignments:
            cluster_assignments[roi_selection] = {}
        cluster_assignments = match_cluster_hash_to_cluster_frame(cluster_frame_dict, cluster_assignments, roi_selection)
        for cluster_cat in cluster_frame_dict[roi_selection].keys():
            unique_clusters = [str(i) for i in
            pd.DataFrame(cluster_frame_dict[roi_selection])[cluster_cat].unique().tolist()]
            if cluster_cat not in ClusterIdentifiers.id_cols and \
                    (check_diff_cluster_subtypes(cluster_assignments, roi_selection, cluster_cat,
                    unique_clusters) or cluster_cat not in cluster_assignments[roi_selection].keys()):
                cluster_assignments[roi_selection][cluster_cat] = {}
                unique_colours = glasbey_palette(len(unique_clusters))
                for clust, colour in zip(unique_clusters, unique_colours):
                    cluster_assignments[roi_selection][cluster_cat][str(clust)] = colour
        return cluster_assignments
    except (KeyError, TypeError):
        return None

def cluster_assignments_from_config(assignments_dict: Union[dict, None]=None,
                                    roi_selection: Union[str, None]=None,
                                    config: Union[dict, None]=None):
    """
    Add the ROI clusters from the db or JSON config to the existing assignments. Prevents
    existing assignments from other ROIs from being overwritten (each ROI has unique cluster assignments)
    """
    if config and roi_selection and 'cluster' in config and config['cluster']:
        assignments_dict = {roi_selection: {}} if not assignments_dict else assignments_dict
        assignments_dict[roi_selection] = config['cluster']
        return assignments_dict
    return dash.no_update


def check_diff_cluster_subtypes(cluster_assignments: dict, roi_selection: str,
                                cluster_cat: str, incoming_subtypes: list):
    """
    Check if the current cluster category has different incoming subtypes that the existing
    dictionary. Could be caused by uploading subsequent cluster assignments with the same category name
    i.e. cluster, but with different subtypes
    """
    if cluster_assignments and roi_selection and cluster_cat and incoming_subtypes and \
        roi_selection in cluster_assignments and cluster_cat in cluster_assignments[roi_selection]:
        current_types = [str(i) for i in list(cluster_assignments[roi_selection][cluster_cat].keys())]
        incoming_subtypes = [str(i) for i in incoming_subtypes]
        # return True if the lists are different so will update
        return set(incoming_subtypes) != set(current_types)
    # by default, process
    return True



def match_cluster_hash_to_cluster_frame(cluster_frame_dict: dict, cluster_assignments: dict,
                                        roi_selection: str) -> dict:
    """
    Match the cluster dictionary entries for a specific ROI to the imported cluster frame.
    Remove any columns in the hash that do not have a corresponding column in the cluster frame
    """
    if cluster_assignments and roi_selection and roi_selection in cluster_assignments:
        for cat in list(cluster_assignments[roi_selection].keys()):
            if cat not in list(cluster_frame_dict[roi_selection].columns):
                del cluster_assignments[roi_selection][cat]
    return cluster_assignments

def cluster_label_children(roi_selection: str=None, cluster_assignments_dict: dict=None,
                           cluster_id_col: str='cluster') -> list:
    """
    Generate the HTML legend for cluster colour assignments used in the side panel
    """
    if None not in (roi_selection, cluster_assignments_dict, cluster_id_col) and \
            roi_selection in cluster_assignments_dict and cluster_id_col in cluster_assignments_dict[roi_selection]:
        children = [html.Span("Cluster assignments\n", style={"color": "black"}), html.Br()]
        for key, value in cluster_assignments_dict[roi_selection][cluster_id_col].items():
            children.append(html.Span(f"{str(key)}\n", style={"color": str(value)}))
            children.append(html.Br())
        return children
    return []

def get_cluster_proj_id_column(cluster_frame: Union[pd.DataFrame, dict]=None) -> Union[str, None]:
    """
    Return the cluster id column name associated with an imported cluster projection frame
    """
    for col in ClusterIdentifiers.id_cols:
        if col in list(pd.DataFrame(cluster_frame).columns):
            return col
    return None

def set_cluster_col_dropdown(cluster_frame: Union[pd.DataFrame, dict]=None) -> Union[list, None]:
    """
    Set the possible cluster projection category options
    """
    cluster_frame = pd.DataFrame(cluster_frame)
    if not cluster_frame.empty and get_cluster_proj_id_column(cluster_frame):
        return [col for col in list(cluster_frame.columns) if \
                col not in ClusterIdentifiers.id_cols]
    return None

def set_default_cluster_col(cluster_frame: dict, roi_selection: str,
                            cur_col: Union[str, None]=None):
    """
    Set the default cluster column when an ROI is selected. By default, return the first column
    in the list, or None if one doesn't exist
    """
    if cluster_frame and roi_selection and roi_selection in cluster_frame and cluster_frame[roi_selection]:
        if cur_col is not None and str(cur_col) in cluster_frame[roi_selection].keys():
            return dash.no_update
        return str(list(cluster_frame[roi_selection].keys())[0])
    return None
