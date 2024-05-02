import os
from ccramic.parsers.roi_parsers import RegionThumbnail
import numpy as np
from ccramic.parsers.pixel_level_parsers import (
    FileParser,
    create_new_blending_dict)
import random

def test_roi_query_parser(get_current_dir):
    mcd = os.path.join(get_current_dir, "query.mcd")
    session_config = {"uploads": [str(mcd)]}
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

    dataset_exclude = "query+++slide0+++PAP_1"
    random.seed(1)
    roi_query = RegionThumbnail(session_config, blend_dict, channels, 3,
                                                     [dataset_exclude]).get_image_dict()
    # assert that the number of queries is 1 less than the total because the current one is excluded
    assert len(roi_query) == 3
    assert dataset_exclude not in roi_query.keys()
    roi_query = RegionThumbnail(session_config, blend_dict, channels, 20, []).get_image_dict()
    assert len(roi_query) == 6

    # # fake mcd returns None
    # fake_mcd = os.path.join(get_current_dir, "fake.txt")
    # session_config = {"uploads": [str(fake_mcd)]}
    # assert RegionThumbnail(session_config, blend_dict, channels, 20, []).get_image_dict() is None

    # assert a key error on an improperly configured session config
    bad_session_config = {"fake_key": [str(mcd)]}
    assert RegionThumbnail(bad_session_config, blend_dict, channels, 20, []).get_image_dict() is None


def test_query_parser_tiff(get_current_dir):
    mcd = os.path.join(get_current_dir, "for_recolour.tiff")
    session_config = {"uploads": [str(mcd)]}
    parse = FileParser(session_config['uploads']).image_dict
    blend_dict = create_new_blending_dict(parse)
    mask_dict = {'for_recolour_mask': {'raw': np.zeros((600, 600)), 'boundary': np.zeros((600, 600, 3))}}
    query_selection = {'names': ['for_recolour']}
    roi_query = RegionThumbnail(session_config, blend_dict, ['channel_1'], 1,
                                dataset_options=list(parse.keys()),
                                mask_dict=mask_dict,
                                predefined_indices=query_selection).get_image_dict()
    assert 'for_recolour+++slide0+++acq0' in roi_query.keys()
    assert len(roi_query) == 1

def test_query_parser_txt(get_current_dir):
    mcd = os.path.join(get_current_dir, "query_from_text.txt")
    session_config = {"uploads": [str(mcd)]}
    parse = FileParser(session_config['uploads']).image_dict
    blend_dict = create_new_blending_dict(parse)
    roi_query = RegionThumbnail(session_config, blend_dict, ['Gd160'], 1,
                                dataset_options=list(parse.keys())).get_image_dict()
    assert 'query_from_text+++slide0+++0' in roi_query.keys()
    assert len(roi_query) == 1

def test_roi_query_parser_predefined(get_current_dir):
    mcd = os.path.join(get_current_dir, "query.mcd")
    session_config = {"uploads": [str(mcd)]}
    dataset_selection = "query+++slide0+++PAP_1"
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
    roi_query = RegionThumbnail(session_config, blend_dict, channels, 4, [],
                predefined_indices=defined_indices, dimension_limit=200).get_image_dict()
    assert len(roi_query) == 2
    assert dataset_selection in roi_query.keys()

    defined_names = {'names': ['PAP_1']}
    roi_query = RegionThumbnail(session_config, blend_dict, channels, 4, [],
                                                     predefined_indices=defined_names).get_image_dict()
    assert len(roi_query) == 1
    assert dataset_selection in roi_query.keys()

    mask_roi_dict = {"PAP_1_mask": {"boundary": np.full((100, 100, 3), 7), "raw": np.full((100, 100), 7)},
                     "HIER_2_mask": {"boundary": np.full((100, 100, 3), 0)},
                     "roi_3_mask": {"boundary": np.zeros((100, 100, 3))}}

    defined_names = {'names': ['PAP_1']}
    query_cell_id_lists = {'PAP_1': [7]}
    roi_query_w_mask = RegionThumbnail(session_config, blend_dict, channels, 4, [],
                            predefined_indices=defined_names, mask_dict=mask_roi_dict,
                                       dataset_options=['query+++slide0+++PAP_1'],
                            query_cell_id_lists=query_cell_id_lists).get_image_dict()
    assert len(roi_query_w_mask) == 1
    assert dataset_selection in roi_query_w_mask.keys()
    assert not np.array_equal(roi_query['query+++slide0+++PAP_1'], roi_query_w_mask['query+++slide0+++PAP_1'])

    # assertion if no query cells are used, just use the boundary
    roi_query_w_mask = RegionThumbnail(session_config, blend_dict, channels, 4, [],
                                       predefined_indices=defined_names, mask_dict=mask_roi_dict,
                                       dataset_options=['query+++slide0+++PAP_1'],
                                       query_cell_id_lists=None).get_image_dict()
    assert len(roi_query_w_mask) == 1
    assert dataset_selection in roi_query_w_mask.keys()
    assert not np.array_equal(roi_query['query+++slide0+++PAP_1'], roi_query_w_mask['query+++slide0+++PAP_1'])

    # assert nothing is returned if the names don't match
    defined_names = {'names': ['PAP_1_mask']}
    query_cell_id_lists = {'PAP_1_mask': [7]}
    roi_query_w_mask = RegionThumbnail(session_config, blend_dict, channels, 4, [],
                                       predefined_indices=defined_names, mask_dict=mask_roi_dict,
                                       dataset_options=['query+++slide0+++PAP_1'],
                                       query_cell_id_lists=query_cell_id_lists).get_image_dict()

    assert not roi_query_w_mask
