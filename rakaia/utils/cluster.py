from typing import Union
import pandas as pd
from dash import html
from rakaia.utils.pixel import glasbey_palette

class ClusterIdentifiers:
    """
    Set the set of mask object id column identifiers permitted for cluster projection
    """
    id_cols = ['cell_id', 'object_id']

def cluster_annotation_frame_import(cur_cluster_dict: dict=None, roi_selection: str=None, cluster_frame:
                                    Union[pd.DataFrame, dict]=None) -> dict:
    """
    Parse the column headers for an imported cluster annotation data frame and verify the presence of at least
    one column identifier
    """
    cur_cluster_dict = {} if not cur_cluster_dict else cur_cluster_dict
    cluster_frame = pd.DataFrame(cluster_frame)
    # for now, use set column names, but expand in the future
    if any([elem in list(cluster_frame.columns) for elem in ClusterIdentifiers.id_cols]) and roi_selection:
        cur_cluster_dict[roi_selection] = cluster_frame
    return cur_cluster_dict if len(cur_cluster_dict) > 0 else None

def assign_colours_to_cluster_annotations(cluster_frame_dict: dict=None, cur_cluster_dict: dict=None,
                                          roi_selection: str=None, cluster_id_col: str='cluster') -> tuple:
    """
    Generate a dictionary of random colours to assign to the clusters for a specific ROI
    cluster frame dict contains the cluster assignments by ROI
    cur_cluster_dict contains current assignments from previous uploads or previous ROIs
    """
    try:
        unique_clusters = pd.DataFrame(cluster_frame_dict[roi_selection])[cluster_id_col].unique().tolist()
        unique_colours = glasbey_palette(len(unique_clusters))
        cluster_assignments = {roi_selection: {}} if not cur_cluster_dict else cur_cluster_dict
        if roi_selection not in cluster_assignments:
            cluster_assignments[roi_selection] = {}
        cluster_assignments = match_cluster_hash_to_cluster_frame(cluster_frame_dict, cluster_assignments, roi_selection)
        if cluster_id_col not in cluster_assignments[roi_selection].keys():
            cluster_assignments[roi_selection][cluster_id_col] = {}
            for clust, colour in zip(unique_clusters, unique_colours):
                cluster_assignments[roi_selection][cluster_id_col][clust] = colour
        return cluster_assignments, list(unique_clusters)
    except (KeyError, TypeError):
        return None, None

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

def set_default_cluster_col(cluster_frame: dict, roi_selection: str):
    """
    Set the default cluster column when an ROI is selected. BY default, return the first column
    in the list, or None if one doesn't exist
    """
    if cluster_frame and roi_selection and roi_selection in cluster_frame and cluster_frame[roi_selection]:
        return str(list(cluster_frame[roi_selection].keys())[0])
    return None
