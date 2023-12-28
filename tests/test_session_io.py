import tempfile
import os
import numpy as np

from ccramic.io.session import (write_blend_config_to_json,
                                write_session_data_to_h5py,
                                subset_mask_for_data_export)

def test_write_config_json():
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_dir = os.path.join(tmpdirname, "fdsdfsdlfkdn", 'downloads')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        blend_dict = {"ArAr80": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Y89": {"color": "#00FF00", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None, "filter_val": None, "filter_sigma": None}, "In113": {"color": "#FF0000", "x_lower_bound": 2, "x_upper_bound": 4, "filter_type": None, "filter_val": None, "filter_sigma": None}, "In115": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Xe131": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Xe134": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Ba136": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "La138": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Pr141": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd142": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd143": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd144": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd145": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd146": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Sm147": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd148": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Sm149": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Nd150": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Eu151": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Sm152": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Eu153": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Sm154": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Gd155": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Gd156": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Gd158": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Tb159": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Gd160": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Dy161": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Dy162": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Dy163": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Dy164": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Ho165": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Er166": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Er167": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Er168": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Tm169": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Er170": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Yb171": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Yb172": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Yb173": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Yb174": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Lu175": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Yb176": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Ir191": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Ir193": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Pt196": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}, "Pb206": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}}
        blend_layers = ["In113", "Y89"]

        global_apply_filter = [" apply/refresh filter"]
        global_filter_type = "gaussian"
        global_filter_val = 3
        global_filter_sigma = 1
        json_path = write_blend_config_to_json(download_dir, blend_dict, blend_layers, global_apply_filter,
                                               global_filter_type, global_filter_val, global_filter_sigma)
        assert os.path.exists(json_path)
        if os.access(json_path, os.W_OK):
            os.remove(json_path)
        assert not os.path.exists(json_path)


def test_write_session_data_to_h5py():

    data_dict = {"roi_1": {"channel_1": np.full((1000, 1000), 1),
                 "channel_2": np.full((1000, 1000), 1),
                "channel_3": np.full((1000, 1000), 1)}}

    blend_dict = {"channel_1": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None,
                                          "filter_type": None, "filter_val": None, "filter_sigma": None},
                  'channel_2': {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None,
                                          "filter_type": None, "filter_val": None, "filter_sigma": None},
                  'channel_3': {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None,
                                          "filter_type": None, "filter_val": None, "filter_sigma": None}}

    metadata_frame = {'channels': list(blend_dict.keys()),
                      'label': list(blend_dict.keys())}
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_dir = os.path.join(tmpdirname, "fdsdfsdlfkdn", 'downloads')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        h5py_path = write_session_data_to_h5py(download_dir, metadata_frame, data_dict, 'roi_1',
                                               blend_dict, None)

        assert os.path.exists(h5py_path)
        if os.access(h5py_path, os.W_OK):
            os.remove(h5py_path)
        assert not os.path.exists(h5py_path)

def test_generate_subset_mask_for_export():
    canvas_layout = {'shapes': [{'line': {'color': 'white', 'width': 2},
                                 'type': 'line', 'x0': 0.875, 'x1': 0.95, 'xref': 'paper', 'y0': 0.05,
                                 'y1': 0.05, 'yref': 'paper'}, {'editable': True,
                                'label': {'text': '', 'texttemplate': ''},
                                'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1,
                                'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                                'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type': 'path',
                                'path': 'M143.55487804878052,146.75609756097563L169.16463414634148,'
                                '144.01219512195124L217.64024390243904,151.32926829268294L235.01829268292684,'
                                '160.47560975609758L243.25,176.02439024390247L243.25,197.06097560975613L235.01829268292684,'
                                '217.18292682926833L219.46951219512198,237.3048780487805L206.66463414634148,'
                                '244.6219512195122L175.56707317073173,247.3658536585366L151.78658536585368,'
                                '245.53658536585365L117.94512195121953,237.3048780487805L107.88414634146342,'
                                '232.7317073170732L94.16463414634147,229.98780487804882L73.1280487804878,'
                                '223.58536585365857L62.15243902439025,209.86585365853662L57.57926829268293,'
                                '197.06097560975613L55.75,193.40243902439028Z'}]}
    mask = subset_mask_for_data_export(canvas_layout, (600, 600))
    assert mask is not None
    assert np.count_nonzero(mask) == 14103

    # assert that an additional shape will increase the number of non zeros
    canvas_layout_2 = {'shapes': [{'line': {'color': 'white', 'width': 2}, 'type': 'line', 'x0': 0.875, 'x1': 0.95, 'xref': 'paper', 'y0': 0.05, 'y1': 0.05, 'yref': 'paper'}, {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4, 'dash': 'solid'}, 'fillcolor': 'rgba(0, 0, 0, 0)', 'fillrule': 'evenodd', 'type': 'path', 'path': 'M143.55487804878052,146.75609756097563L169.16463414634148,144.01219512195124L217.64024390243904,151.32926829268294L235.01829268292684,160.47560975609758L243.25,176.02439024390247L243.25,197.06097560975613L235.01829268292684,217.18292682926833L219.46951219512198,237.3048780487805L206.66463414634148,244.6219512195122L175.56707317073173,247.3658536585366L151.78658536585368,245.53658536585365L117.94512195121953,237.3048780487805L107.88414634146342,232.7317073170732L94.16463414634147,229.98780487804882L73.1280487804878,223.58536585365857L62.15243902439025,209.86585365853662L57.57926829268293,197.06097560975613L55.75,193.40243902439028Z'}, {'editable': True, 'label': {'text': '', 'texttemplate': ''}, 'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4, 'dash': 'solid'}, 'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type': 'path', 'path': 'M372.2134146341464,265.6585365853659L388.6768292682927,234.5609756097561L401.4817073170732,197.97560975609758L407.8841463414634,192.4878048780488L425.2621951219512,194.31707317073173L438.0670731707317,202.5487804878049L449.0426829268293,215.35365853658539L449.95731707317077,231.81707317073173L470.07926829268297,241.8780487804878L501.17682926829275,249.1951219512195L507.57926829268297,257.4268292682927L512.1524390243903,275.719512195122L521.298780487805,339.7439024390244L518.5548780487806,350.719512195122L512.1524390243903,361.6951219512195L503.92073170731715,368.0975609756098L474.65243902439033,375.4146341463415L451.7865853658537,368.0975609756098L433.4939024390244,350.719512195122L427.0914634146342,354.3780487804878L413.3719512195122,363.5243902439025L400.5670731707317,369.0121951219512L385.9329268292683,376.3292682926829L371.2987804878049,373.5853658536586L356.6646341463415,364.4390243902439L339.2865853658537,342.4878048780488L328.3109756097561,325.109756097561L316.4207317073171,307.7317073170732L316.4207317073171,303.1585365853659L325.5670731707317,301.3292682926829Z'}]}
    mask_2 = subset_mask_for_data_export(canvas_layout_2, (600, 600))
    assert mask_2 is not None
    assert np.count_nonzero(mask_2) > 14103

    canvas_layout = {}
    assert subset_mask_for_data_export(canvas_layout, (600, 600)) is None

    canvas_layout = {'shapes': [{'line': {'color': 'white', 'width': 2}, 'type': 'line', 'x0': 0.875, 'x1': 0.95, 'xref': 'paper', 'y0': 0.05, 'y1': 0.05, 'yref': 'paper'}, {'editable': True, 'label': {'text': '', 'texttemplate': ''}, 'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4, 'dash': 'solid'}, 'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type': 'rect', 'x0': 305.4451219512195, 'y0': 87.76219512195122, 'x1': 515.8109756097562, 'y1': 346.6036585365854}]}
    assert subset_mask_for_data_export(canvas_layout, (600, 600)) is None
