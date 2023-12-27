import tempfile
import os
import numpy as np

from ccramic.io.session import write_blend_config_to_json, write_session_data_to_h5py

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
