import os
from ccramic.parsers.roi_parsers import generate_multi_roi_images_from_query
import numpy as np

def test_roi_query_parser(get_current_dir):
    mcd = os.path.join(get_current_dir, "query.mcd")
    session_config = {"uploads": [str(mcd)]}
    dataset_selection = "PAP"
    channels = ["Ir191", "Ir193"]
    blend_dict = {"ArAr80": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None,
                             "filter_val": None},
                  "Sn120": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe126": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "I127": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe128": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe131": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe134": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Ba138": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Ir191": {"color": "#FF0000", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None, "filter_val": None},
                  "Ir193": {"color": "#FF0000", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None, "filter_val": None},
                  "Pb208": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None}}

    roi_query = generate_multi_roi_images_from_query(session_config, blend_dict, channels, 4,
                                                     [dataset_selection])
    assert len(roi_query) == 4
    assert dataset_selection not in roi_query.keys()
    roi_query = generate_multi_roi_images_from_query(session_config, blend_dict, channels, 20, [])
    assert len(roi_query) == 6


def test_roi_query_parser_predefined(get_current_dir):
    mcd = os.path.join(get_current_dir, "query.mcd")
    session_config = {"uploads": [str(mcd)]}
    dataset_selection = "query+++slide0+++PAP"
    channels = ["Ir191", "Ir193"]
    blend_dict = {"ArAr80": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None,
                             "filter_val": None},
                  "Sn120": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe126": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "I127": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe128": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe131": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Xe134": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Ba138": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None},
                  "Ir191": {"color": "#FF0000", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None, "filter_val": None},
                  "Ir193": {"color": "#FF0000", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None, "filter_val": None},
                  "Pb208": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None}}


    defined_indices = {'indices': [0, 1]}
    roi_query = generate_multi_roi_images_from_query(session_config, blend_dict, channels, 4, [],
                                                     predefined_indices=defined_indices)
    assert len(roi_query) == 2
    assert dataset_selection in roi_query.keys()

    defined_names = {'names': ['PAP']}
    roi_query = generate_multi_roi_images_from_query(session_config, blend_dict, channels, 4, [],
                                                     predefined_indices=defined_names)
    assert len(roi_query) == 1
    assert dataset_selection in roi_query.keys()

    mask_roi_dict = {"PAP": {"boundary": np.full((100, 100), 7), "raw": np.full((100, 100), 7)},
                     "HIER": {"boundary": np.full((100, 100), 0)},
                     "roi_3": {"boundary": np.zeros((100, 100))}}

    defined_names = {'names': ['PAP']}
    query_cell_id_lists = {'PAP': [7]}
    roi_query_w_mask = generate_multi_roi_images_from_query(session_config, blend_dict, channels, 4, [],
                            predefined_indices=defined_names, mask_dict=mask_roi_dict, dataset_options=None,
                            query_cell_id_lists=query_cell_id_lists)
    assert len(roi_query_w_mask) == 1
    assert dataset_selection in roi_query_w_mask.keys()
    assert not np.array_equal(roi_query['query+++slide0+++PAP'], roi_query_w_mask['query+++slide0+++PAP'])
