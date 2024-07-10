import pandas as pd
from dash import html
from rakaia.utils.pixel import (
    glasbey_palette)

def assign_colours_to_cluster_annotations(cluster_frame_dict: dict=None, cur_cluster_dict: dict=None,
                                          roi_selection: str=None, cluster_id_col: str='cluster'):
    """
    Generate a dictionary of random colours to assign to the clusters for a specific ROI
    cluster frame dict contains the cluster assignments by ROI
    cur_cluster_dict contains current assignments from previous uploads or previous ROIs
    """
    try:
        unique_clusters = pd.DataFrame(cluster_frame_dict[roi_selection])[cluster_id_col].unique().tolist()
        unique_colours = glasbey_palette(len(unique_clusters))
        cluster_assignments = {roi_selection: {}} if not cur_cluster_dict else cur_cluster_dict
        # reset the dictionary if either the ROI doesn't have clusters, or new assignments are uploaded
        if (roi_selection not in cluster_assignments) or (roi_selection in cluster_assignments and
            len(unique_clusters) != len(cluster_assignments[roi_selection])):
            cluster_assignments[roi_selection] = {}
            for clust, colour in zip(unique_clusters, unique_colours):
                cluster_assignments[roi_selection][clust] = colour
        return cluster_assignments, list(unique_clusters)
    except (KeyError, TypeError):
        return None, None

def cluster_label_children(roi_selection: str=None, cluster_assignments_dict: dict=None):
    """
    Generate the HTML legend for cluster colour assignments used in the side panel
    """
    if None not in (roi_selection, cluster_assignments_dict) and roi_selection in cluster_assignments_dict:
        children = [html.Span("Cluster assignments\n", style={"color": "black"}), html.Br()]
        for key, value in cluster_assignments_dict[roi_selection].items():
            children.append(html.Span(f"{str(key)}\n", style={"color": str(value)}))
            children.append(html.Br())
        return children
    return []
