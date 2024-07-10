import dash
import pandas as pd
from rakaia.utils.cluster import (
    assign_colours_to_cluster_annotations,
    cluster_label_children)
from dash import html

def test_basic_cluster_colour_assignments():
    cluster_frame = pd.DataFrame(
        {"cell_id": range(1, 101),
         "cluster": ["cell_type_1"] * 25 + ["cell_type_2"] * 25 + ["cell_type_3"] * 25 + ["cell_type_4"] * 25,
         })
    clust_dict = {"roi_1": cluster_frame}
    cur_cluster_dict = None
    colours, options = assign_colours_to_cluster_annotations(clust_dict, cur_cluster_dict, "roi_1")
    assert len(colours['roi_1']) == len(options) == 4
    cluster_frame_2 = pd.DataFrame(
        {"cell_id": range(1, 201),
         "cluster": ["cell_type_1"] * 40 + ["cell_type_2"] * 40 + ["cell_type_3"] * 40 +
                    ["cell_type_4"] * 40 + ["cell_type_5"] * 40,
         })
    clust_dict_2 = {"roi_1": cluster_frame_2}
    colours, options = assign_colours_to_cluster_annotations(clust_dict_2, clust_dict, "roi_1")
    assert len(colours['roi_1']) == len(options) == 5

    assert assign_colours_to_cluster_annotations(clust_dict_2, clust_dict, "roi_2") == (None, None)

    sidebar_labels = cluster_label_children("roi_1", colours)
    assert isinstance(sidebar_labels, list)
    assert isinstance(sidebar_labels[0], html.Span)
    assert len(sidebar_labels) == (2 * len(colours["roi_1"])) + 2

    assert len(cluster_label_children()) == 0
