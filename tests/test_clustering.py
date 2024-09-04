import os
import random
import pandas as pd
from pandas.testing import assert_frame_equal
from dash import html
from rakaia.parsers.object import parse_quantification_sheet_from_h5ad
from rakaia.utils.cluster import (
    assign_colours_to_cluster_annotations,
    cluster_label_children,
    cluster_annotation_frame_import,
    get_cluster_proj_id_column,
    set_cluster_col_dropdown,
    match_cluster_hash_to_cluster_frame,
    set_default_cluster_col,
    QuantificationClusterMerge,
    subset_cluster_frame)

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
         "cell_type": ["cell_type_5"] * 25 + ["cell_type_6"] * 25 + ["cell_type_7"] * 25 + ["cell_type_8"] * 25,
         })
    clust_dict = {"roi_1": cluster_frame}
    cur_cluster_dict = None
    colours = assign_colours_to_cluster_annotations(clust_dict, cur_cluster_dict, "roi_1")
    assert len(colours['roi_1']) == 2
    assert len(colours['roi_1']['cluster']) == 4
    cluster_frame_2 = pd.DataFrame(
        {"cell_id": range(1, 201),
         "cluster": ["cell_type_1"] * 40 + ["cell_type_2"] * 40 + ["cell_type_3"] * 40 +
                    ["cell_type_4"] * 40 + ["cell_type_5"] * 40,
         })
    clust_dict_2 = {"roi_1": cluster_frame_2}
    colours = assign_colours_to_cluster_annotations(clust_dict_2, {"roi_1": {}}, "roi_1")
    assert len(colours['roi_1']['cluster']) == 5

    assert assign_colours_to_cluster_annotations(clust_dict_2, clust_dict, "roi_2") is None

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

def test_cluster_frame_subsetting(get_current_dir):
    cluster_assignments = {"roi_1": pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))}
    subset = subset_cluster_frame(cluster_assignments, "roi_1", "cluster", ['Type_1', 'Type_2'])
    assert len(subset) < len(pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv")))
    assert isinstance(subset, pd.DataFrame)
    subset_2 = subset_cluster_frame(cluster_assignments, "roi_2", "cluster", ['Type_1', 'Type_2'])
    assert isinstance(subset_2, dict)

    objs = random.sample(range(100, 999), 25)
    subset_w_objs = subset_cluster_frame(cluster_assignments, "roi_1", "cluster", ['Type_1', 'Type_2'], objs)
    assert len(subset_w_objs) == len(objs)


def test_default_cluster_col():
    type_cols = {"roi_1": {"cat_1": {"one": "blue"}, "cluster": {"Type_1": "blue"}}}
    assert set_default_cluster_col(type_cols, "roi_1") == "cat_1"
    assert set_default_cluster_col({"roi_1": {}}, "roi_1") is None
    assert set_default_cluster_col({}, "roi_1") is None
    assert set_default_cluster_col({"other": {"cat_1": {"one": "red"}}}, "roi_1") is None

def test_quant_to_cluster_transfer(get_current_dir):
    data_selection = "test---slide0---chr10-h54h54-Gd158_2_18"
    measurements = parse_quantification_sheet_from_h5ad((os.path.join(get_current_dir, "from_steinbock.h5ad")))
    clust_out = QuantificationClusterMerge(measurements, data_selection,
                                              "description", None, delimiter="---").get_cluster_frame()
    merged_clust = clust_out[data_selection]
    assert all(elem in merged_clust.columns for elem in ['cell_id', 'description'])
    merged_clust['cell_type'] = "immune"
    merged_clust = merged_clust.drop(['description'], axis=1)
    clust_out[data_selection] = merged_clust
    out_clust_2 = QuantificationClusterMerge(measurements, data_selection, "description",
                                                clust_out, delimiter="---").get_cluster_frame()
    merged_clust_2 = out_clust_2[data_selection]
    assert all(elem in merged_clust_2.columns for elem in ['cell_id', 'description', 'cell_type'])

    # using sample and old pipeline syntax
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    measurements['cluster'] = 'epithelial'
    # old pipeline: rois take the form of filename_index
    dataset_options = ["test+++slide0+++test_1", "test+++slide0+++test_2"]
    out_clust_3 = QuantificationClusterMerge(measurements, dataset_options[0], "cluster",
                    None, '+++', None, dataset_options).get_cluster_frame()
    assert dataset_options[0] in out_clust_3.keys()
    assert all(elem in out_clust_3[dataset_options[0]].columns for elem in ['cell_id', 'cluster'])
