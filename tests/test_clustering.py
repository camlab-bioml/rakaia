import os
import pandas as pd
from pandas.testing import assert_frame_equal
from rakaia.utils.cluster import (
    assign_colours_to_cluster_annotations,
    cluster_label_children,
    cluster_annotation_frame_import,
    get_cluster_proj_id_column,
    set_cluster_col_dropdown,
    match_cluster_hash_to_cluster_frame,
    set_default_cluster_col)
from dash import html

def test_basic_colparse_cluster(get_current_dir):
    cluster_assignments = pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))
    assert get_cluster_proj_id_column(cluster_assignments) == "object_id"
    fake_frame = {"identifier": [1, 2, 3, 4, 5]}
    assert not get_cluster_proj_id_column(fake_frame)
    assert set_cluster_col_dropdown(cluster_assignments) == ['cluster']
    assert not set_cluster_col_dropdown(fake_frame)
def test_populating_cluster_annotation_dict():
    cluster_frame = pd.DataFrame({"cell_id": [1, 2, 3, 4, 5],
                                 "cluster": ["immune"] * 5})
    session_cluster_dict = cluster_annotation_frame_import(None, "roi_1", cluster_frame)
    assert_frame_equal(cluster_frame, session_cluster_dict['roi_1'])
    malformed = pd.DataFrame({"col_1": [1, 2, 3, 4, 5],
                                  "col_2": ["immune"] * 5})
    session_cluster_dict = cluster_annotation_frame_import(session_cluster_dict, "roi_2", malformed)
    assert "roi_2" not in session_cluster_dict.keys()
    assert "roi_1" in session_cluster_dict.keys()

def test_basic_cluster_colour_assignments():
    cluster_frame = pd.DataFrame(
        {"cell_id": range(1, 101),
         "cluster": ["cell_type_1"] * 25 + ["cell_type_2"] * 25 + ["cell_type_3"] * 25 + ["cell_type_4"] * 25,
         })
    clust_dict = {"roi_1": cluster_frame}
    cur_cluster_dict = None
    colours, options = assign_colours_to_cluster_annotations(clust_dict, cur_cluster_dict, "roi_1")
    assert len(colours['roi_1']['cluster']) == len(options) == 4
    cluster_frame_2 = pd.DataFrame(
        {"cell_id": range(1, 201),
         "cluster": ["cell_type_1"] * 40 + ["cell_type_2"] * 40 + ["cell_type_3"] * 40 +
                    ["cell_type_4"] * 40 + ["cell_type_5"] * 40,
         })
    clust_dict_2 = {"roi_1": cluster_frame_2}
    colours, options = assign_colours_to_cluster_annotations(clust_dict_2, {"roi_1": {}}, "roi_1")
    assert len(colours['roi_1']['cluster']) == len(options) == 5

    assert assign_colours_to_cluster_annotations(clust_dict_2, clust_dict, "roi_2") == (None, None)

    sidebar_labels = cluster_label_children("roi_1", colours)
    assert isinstance(sidebar_labels, list)
    assert isinstance(sidebar_labels[0], html.Span)
    assert len(sidebar_labels) == (2 * len(colours["roi_1"]['cluster'])) + 2

    assert len(cluster_label_children()) == 0


def test_matching_frame_to_hash(get_current_dir):
    cluster_assignments = {"roi_1": pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))}
    type_cols = {"roi_1": {"fake": {"one": "blue"}, "cluster": {"Type_1": "blue"}}}
    cleaned_cols = match_cluster_hash_to_cluster_frame(cluster_assignments, type_cols, "roi_1")
    assert 'fake' not in cleaned_cols['roi_1'].keys()
    assert 'cluster' in cleaned_cols['roi_1'].keys()

def test_default_cluster_col():
    type_cols = {"roi_1": {"cat_1": {"one": "blue"}, "cluster": {"Type_1": "blue"}}}
    assert set_default_cluster_col(type_cols, "roi_1") == "cat_1"
    assert set_default_cluster_col({"roi_1": {}}, "roi_1") is None
    assert set_default_cluster_col({}, "roi_1") is None
    assert set_default_cluster_col({"other": {"cat_1": {"one": "red"}}}, "roi_1") is None
