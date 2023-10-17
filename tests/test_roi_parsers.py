from ccramic.app.parsers.roi_parsers import *

def test_roi_query_parser(get_current_dir):
    mcd = os.path.join(get_current_dir, "query.mcd")
    session_config = {"uploads": [str(mcd)]}
    dataset_selection = "DotBlot_3+++slide0+++PAP"
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

    roi_query = generate_multi_roi_images_from_query(dataset_selection, session_config, blend_dict, channels, 4)
    assert len(roi_query) == 4
    assert dataset_selection not in roi_query.keys()
    roi_query = generate_multi_roi_images_from_query(dataset_selection, session_config, blend_dict, channels, 20)
    assert len(roi_query) == 6
