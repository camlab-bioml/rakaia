import os
import dash
import pytest
import plotly
from dash.exceptions import PreventUpdate
import pandas as pd
from rakaia.parsers.pixel import create_new_blending_dict
from PIL import Image
import numpy as np
from rakaia.utils.pixel import (
    recolour_greyscale,
    apply_preset_to_array,
    resize_for_canvas,
    filter_by_upper_and_lower_bound,
    pixel_hist_from_array,
    get_default_channel_upper_bound_by_percentile,
    validate_incoming_metadata_table,
    get_all_images_by_channel_name,
    per_channel_intensity_hovertext,
    create_new_coord_bounds,
    get_area_statistics_from_rect,
    delete_dataset_option_from_list_interactively,
    set_channel_list_order,
    path_to_indices,
    path_to_mask,
    get_area_statistics_from_closed_path,
    get_bounding_box_for_svgpath,
    select_random_colour_for_channel,
    apply_preset_to_blend_dict,
    is_rgb_color,
    generate_default_swatches,
    random_hex_colour_generator,
    get_additive_image,
    get_first_image_from_roi_dictionary,
    set_array_storage_type_from_config,
    apply_filter_to_array,
    split_string_at_pattern,
    no_filter_chosen,
    channel_filter_matches,
    ag_grid_cell_styling_conditions,
    MarkerCorrelation, high_low_values_from_zoom_layout,
    glasbey_palette,
    layers_exist,
    add_saved_blend)

def test_string_splitting():
    exp, slide, acq = split_string_at_pattern("+exp1++++slide0+++acq1")
    assert acq == "acq1"

def test_layer_condition():
    assert layers_exist({"roi_1": {"channel_1": None}}, "roi_1")
    assert not layers_exist({"roi_1": {}}, "roi_1")
    assert not layers_exist({"roi_1": {"channel_1": None}}, "roi_2")

def test_identify_rgb_codes():
    assert is_rgb_color('#FAF0E6')
    assert not is_rgb_color('#FAF0')
    assert not is_rgb_color('#NotRgb')
    assert not is_rgb_color('FAF0E6')

def test_return_array_dtype():
    assert str(set_array_storage_type_from_config()) == "<class 'numpy.float32'>"
    assert str(set_array_storage_type_from_config("int")) == "<class 'numpy.uint16'>"
    with pytest.raises(TypeError):
        set_array_storage_type_from_config("fake_type")

def test_glasbey_palette():
    palette = glasbey_palette()
    assert len(set(palette)) == len(palette)
    palette_longer = glasbey_palette(20)
    assert palette_longer[0:len(palette)] == palette

def test_random_hex_colour_generator():
    random_cols = random_hex_colour_generator()
    assert len(random_cols) == 10
    assert len(set(random_cols)) == 10

    random_cols = random_hex_colour_generator(75)
    assert len(random_cols) == 75
    assert len(set(random_cols)) == 75
    assert all([len(elem) == 7 for elem in random_cols])

def test_generate_default_swatches():
    config = {'swatches': ["#0000FF", "#0000FF", "#0000FF", "#0000FF"]}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 4
    config = {'swatches': "#0000FF,#0000FF,#0000FF,#0000FF"}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 4


    config = {'swatches': ["#0000FF", "#0000FF", "fake", "#0000FF"]}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 3
    config = {'swatches': "#0000FF,fake,#0000FF,#0000FF"}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 3

    config = {'swatches': []}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 7

    config = {'swatches': "None"}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 7

    config = {'fake_key': "None"}
    swatches = generate_default_swatches(config)
    assert len(swatches) == 7

def test_basic_recolour_non_white(get_current_dir):
    greyscale = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    as_rgb = greyscale.convert('RGB')
    pixels = as_rgb.load()
    for i in range(greyscale.height):
        for j in range(greyscale.width):
            assert len(tuple(set(pixels[i, j]))) == 1

    recoloured = recolour_greyscale(np.array(greyscale), colour='#D14A1A')
    recoloured_image = Image.fromarray(recoloured)
    recoloured_pixels = recoloured_image.load()
    assert recoloured_image.height == greyscale.height
    assert recoloured_image.width == greyscale.width
    for i in range(recoloured_image.height):
        for j in range(greyscale.width):
            assert len(tuple(set(recoloured_pixels[i, j]))) >= 1
            if recoloured_pixels[i, j] != (0, 0, 0):
                assert recoloured_pixels[i, j] != pixels[i, j]


def test_basic_recolour_white(get_current_dir):
    greyscale = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = Image.fromarray(np.array(greyscale).astype(np.uint8))
    as_rgb = greyscale.convert('RGB')
    pixels = as_rgb.load()
    for i in range(greyscale.height):
        for j in range(greyscale.width):
            assert len(tuple(set(pixels[i, j]))) == 1

    recoloured = recolour_greyscale(np.array(greyscale).astype(np.uint8), colour='#FFFFFF')
    assert isinstance(recoloured, np.ndarray)
    recoloured_image = Image.fromarray(recoloured)
    recoloured_pixels = recoloured_image.load()
    assert recoloured_image.height == greyscale.height
    assert recoloured_image.width == greyscale.width
    for i in range(recoloured_image.height):
        for j in range(greyscale.width):
            assert len(tuple(set(recoloured_pixels[i, j]))) == 1
            assert recoloured_pixels[i, j] == pixels[i, j]

def test_median_gaussian_filtering(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale_image = np.array(greyscale_image)
    image = np.array(Image.fromarray(greyscale_image).convert('RGB'))
    blend_dict = {'color': '#BE4115',
     'x_lower_bound': None,
     'x_upper_bound': None,
     'filter_type': 'gaussian',
     'filter_val': 5,
        'filter_sigma': 1}
    gaussian = apply_preset_to_array(image, blend_dict)
    assert np.mean(image) != np.mean(gaussian)

    # assert that nothing happens if the gaussian filter is even
    blend_dict_2 = {'color': '#BE4115',
                  'x_lower_bound': None,
                  'x_upper_bound': None,
                  'filter_type': 'gaussian',
                  'filter_val': 4,
                  'filter_sigma': 1}
    gaussian_2 = apply_preset_to_array(image, blend_dict_2)
    assert np.mean(image) == np.mean(gaussian_2)

    blend_dict_3 = {'color': '#BE4115',
                    'x_lower_bound': None,
                    'x_upper_bound': None,
                    'filter_type': 'median',
                    'filter_val': 3,
                    'filter_sigma': 1}
    median_1 = apply_preset_to_array(image, blend_dict_3)
    assert np.mean(image) != np.mean(median_1)

    # assert nothing happens if the median value is negative
    blend_dict_4 = {'color': '#BE4115',
                    'x_lower_bound': None,
                    'x_upper_bound': None,
                    'filter_type': 'median',
                    'filter_val': -1,
                    'filter_sigma': 1}
    median_2 = apply_preset_to_array(image, blend_dict_4)
    assert np.mean(image) == np.mean(median_2)


def test_resize_canvas_image(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = Image.fromarray(np.array(greyscale_image))
    resized = resize_for_canvas(greyscale)
    assert resized.shape[0] == 400 == resized.shape[1]

    resized_different_size = resize_for_canvas(greyscale_image, basewidth=666)
    assert isinstance(resized_different_size, np.ndarray)

    assert resized_different_size.shape[0] == 666 == resized_different_size.shape[1]


def test_filtering_intensity_changes(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    filtered_1 = filter_by_upper_and_lower_bound(greyscale, lower_bound=51, upper_bound=450)
    original_pixels = Image.fromarray(greyscale).load()
    new_pixels = Image.fromarray(filtered_1).load()
    assert np.max(filtered_1) == 255
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            if original_pixels[i, j] < 51:
                assert new_pixels[i, j] == 0
            else:
                # if the priginal pixel is 52 or more intense, the final value will be at least the scale value
                if original_pixels[i, j] >= 52:
                    assert new_pixels[i, j] >= (255 / (450 - 51))
                else:
                    # otherwise, the original pixel is either 0 or a fraction of the scale value (likely a float)
                    assert 0 <= new_pixels[i, j] < (255 / (450 - 51))

    assert np.max(greyscale) >= np.max(filtered_1)


def test_filtering_intensity_changes_none(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    filtered_1 = filter_by_upper_and_lower_bound(greyscale, lower_bound=None, upper_bound=None)
    original_pixels = Image.fromarray(greyscale).load()
    new_pixels = Image.fromarray(filtered_1).load()
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            assert int(original_pixels[i, j]) == int(new_pixels[i, j])

    assert int(np.max(greyscale)) == int(np.max(filtered_1))


def test_filtering_intensity_changes_low(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    filtered_1 = filter_by_upper_and_lower_bound(greyscale, lower_bound=1, upper_bound=15)
    original_pixels = Image.fromarray(greyscale).load()
    new_pixels = Image.fromarray(filtered_1).load()
    # assert that when a low upper bound is used, it scales up to the max of 255
    assert int(round(np.max(filtered_1))) == 255
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            if original_pixels[i, j] < 1:
                assert new_pixels[i, j] == 0
            else:
                assert new_pixels[i, j] <= 255

    assert np.max(greyscale) >= np.max(filtered_1)

def test_filtering_intensity_changes_same_vals(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    filtered_1 = filter_by_upper_and_lower_bound(greyscale, lower_bound=100, upper_bound=100)
    original_pixels = Image.fromarray(greyscale).load()
    new_pixels = Image.fromarray(filtered_1).load()
    # assert that when a low upper bound is used, it scales up to the max of 255
    assert int(round(np.max(filtered_1))) == 0
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            if original_pixels[i, j] < 1:
                assert new_pixels[i, j] == 0
    assert np.max(greyscale) >= np.max(filtered_1)

def test_generate_histogram(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    histogram, array_max = pixel_hist_from_array(greyscale)
    assert isinstance(histogram, plotly.graph_objs._figure.Figure)
    assert histogram["data"] is not None
    assert histogram["layout"] is not None
    values = histogram["data"][0]['x']
    assert int(max(values)) == int(array_max)
    assert len(values) == 360001
    assert array_max == np.max(greyscale)

    larger_array = np.full((2000, 2000), 10)
    larger_array[10, 10] = 56487
    histogram, array_max = pixel_hist_from_array(larger_array)
    assert isinstance(histogram, plotly.graph_objs._figure.Figure)
    assert histogram["data"] is not None
    assert histogram["layout"] is not None
    values = histogram["data"][0]['x']
    assert int(max(values)) == int(array_max)
    assert len(values) == 1000001
    assert array_max == 56487


def test_basic_blend_dict_params():

    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                       "Nuclear": np.array([1, 1, 1, 1]),
                                                       "Cytoplasm": np.array([2, 2, 2, 2])},
                    "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                       "Nuclear": np.array([4, 4, 4, 4]),
                                                       "Cytoplasm": np.array([5, 5, 5, 5])}
                                              }

    blend_dict = create_new_blending_dict(upload_dict)
    assert len(upload_dict.keys()) == 2
    for roi in upload_dict.keys():
        assert len(upload_dict[roi].keys()) == 3


    blend_dict["DNA"] = {'color': '#BE4115',
                                                          'x_lower_bound': 200,
                                                          'x_upper_bound': 1000,
                                                          'y_ceiling': 12500,
                                                          'filter_type': 'gaussian',
                                                          'filter_val': 5}

    blend_dict["Nuclear"] = {'color': '#15BEB0',
                                                          'x_lower_bound': 0.25,
                                                          'x_upper_bound': 55,
                                                          'y_ceiling': 67000,
                                                          'filter_type': 'median',
                                                          'filter_val': 3}

    blend_dict["Cytoplasm"] = {'color': '#BA15BE',
                                                          'x_lower_bound': -0.4,
                                                          'x_upper_bound': 14,
                                                          'y_ceiling': 900,
                                                          'filter_type': 'median',
                                                          'filter_val': 7}

    # check that the default values are either hex white or None
    possibilities = ['#FFFFFF', None]

    # assert that the first ROI has non default params
    for channel in blend_dict.keys():
        for blend_param in blend_dict[channel].values():
            assert blend_param not in possibilities

def test_basic_preset_apply_blend_dict():
    blend_dict = {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}
    preset_dict = {"color": "#FF00FF", "x_lower_bound": 1, "x_upper_bound": 100, "filter_type": "median",
                   "filter_val": 3, "filter_sigma": 1}
    blend_dict = apply_preset_to_blend_dict(blend_dict, preset_dict)
    for key, value in blend_dict.items():
        if key != 'color':
            assert value == preset_dict[key]

def test_calculate_percentile_intensity(get_current_dir):
    array = np.array(Image.open(os.path.join(get_current_dir, "for_recolour.tiff")))
    default_val = get_default_channel_upper_bound_by_percentile(array=array)
    assert default_val == 70.67082221984863
    lower_val = get_default_channel_upper_bound_by_percentile(array=array, percentile=50)
    assert lower_val == 10.202707290649414
    assert default_val > lower_val
    assert get_default_channel_upper_bound_by_percentile(array=array, percentile=100) == float(np.max(array))
    assert get_default_channel_upper_bound_by_percentile(array=array, percentile=100) > \
           get_default_channel_upper_bound_by_percentile(array=array, percentile=99.999)


def test_default_min_percentile_scaling():
    white = np.zeros((512,512), 'uint8')
    assert get_default_channel_upper_bound_by_percentile(white) == 1.0

def test_validation_of_channel_metadata(get_current_dir):
    metadata = pd.read_csv(os.path.join(get_current_dir, "channel_metadata.csv"))
    upload_dict = {"roi_1": {}, 'metadata': None, 'metadata_columns': None}
    for channel in metadata['Channel Name'].to_list():
        upload_dict["roi_1"][channel] = None
    metadata_import = pd.DataFrame(validate_incoming_metadata_table(metadata, upload_dict))
    assert len(metadata_import) == len(metadata) == 36
    empty_dict = {"roi_1": {}, 'metadata': None, 'metadata_columns': None}
    assert validate_incoming_metadata_table(metadata, empty_dict) is None

    # require the properly named column to exist
    empty_frame = pd.DataFrame({'Random': metadata['Channel Label'].to_list()})
    assert validate_incoming_metadata_table(empty_frame, upload_dict) is None

    # empty frame doesn't work
    empty_frame = pd.DataFrame({"Channel Label": []})
    assert validate_incoming_metadata_table(empty_frame, upload_dict) is None

def test_acquire_acq_images_for_gallery():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                   "Nuclear": np.array([1, 1, 1, 1]),
                                                   "Cytoplasm": np.array([2, 2, 2, 2])},
                   "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                   "Nuclear": np.array([4, 4, 4, 4]),
                                                   "Cytoplasm": np.array([5, 5, 5, 5])}
                   }
    assert len(get_all_images_by_channel_name(upload_dict, "DNA")) == 2
    assert len(get_all_images_by_channel_name(upload_dict, "fake")) == 0

def test_make_hovertext_from_channel_list():
    channel_list = ["channel_1", "channel_2"]
    assert per_channel_intensity_hovertext(channel_list) == "x: %{x}, y: %{y} <br>channel_1: %{customdata[0]} " \
                                                            "<br>channel_2: " \
                                                            "%{customdata[1]} <br><extra></extra>"

    # assert that the default hover text is used if a malformed channel input is passed
    assert per_channel_intensity_hovertext("not_a_list") == "x: %{x}, y: %{y} <br><extra></extra>"


def test_coord_navigation():
    window_dict = {"x_high": 200, "x_low": 100, "y_high": 200, "y_low": 100}
    new_coords = create_new_coord_bounds(window_dict, 500, 500)
    assert new_coords[0] == 450.0
    assert new_coords[1] == 550.0
    assert new_coords[1] - new_coords[0] == 100.0

    window_dict = {"x_high": 50, "x_low": 100, "y_high": 10}
    assert create_new_coord_bounds(window_dict, 500, 500) is None


def test_get_statistics_from_rect_array(get_current_dir):
    array = np.array(Image.open(os.path.join(get_current_dir, "for_recolour.tiff")))
    stats_1 = get_area_statistics_from_rect(array, 100, 200, 100, 200)
    assert 27.55 <= stats_1[0] <= 27.57
    bad_stats = get_area_statistics_from_rect(array, 10000, 20000, 100, 200)
    assert all([elem is None for elem in bad_stats])

def test_basic_dataset_dropdown_removal():
    dataset_options = ["dataset1", "dataset2"]
    removed = delete_dataset_option_from_list_interactively(1, "dataset2", dataset_options)
    assert "dataset2" not in removed[0]
    assert "dataset1" in removed[0]
    assert isinstance(removed[-1], dash._callback.NoUpdate)
    with pytest.raises(PreventUpdate):
        delete_dataset_option_from_list_interactively(1, None, dataset_options)
    with pytest.raises(PreventUpdate):
        delete_dataset_option_from_list_interactively(0, "dataset2", dataset_options)

def test_basic_channel_ordering():
    rowdata = [{"Channel": "channel_2"}, {"Channel": "channel_1"},
               {"Channel": "channel_3"}]
    current_blend = ["channel_1", "channel_3", "channel_2"]
    aliases = {"channel_1": "channel_1", "channel_3": "channel_3", "channel_2": "channel_2"}
    channel_order = set_channel_list_order(1, rowdata, None, current_blend, aliases, "image_layers")
    assert channel_order == ['channel_1', 'channel_3', 'channel_2']
    channel_order = set_channel_list_order(1, rowdata, None, current_blend, aliases, "set-sort")
    assert channel_order == ['channel_2', 'channel_1', 'channel_3']
    current_blend = ["channel_1", "channel_3", "channel_2", "channel_4"]
    channel_order = set_channel_list_order(1, rowdata, channel_order, current_blend, aliases, "image_layers")
    assert channel_order == ['channel_2', 'channel_1', 'channel_3', 'channel_4']
    channel_order = set_channel_list_order(1, rowdata, channel_order, [], aliases, "image_layers")
    assert len(channel_order) == 0

def test_basic_svgpath_pixel_mask():
    array = np.zeros((600, 600))
    svgpath = "M222.86561906127866,131.26498798809564L232.88973251102587,145.5851500591631L235.75376492523935," \
              "151.8860213704328L235.467361683818,158.18689268170246L231.74411954534048,161.33732833733728L224.58403850980676," \
              "162.4829413030227L212.84150561153143,161.33732833733728L204.82221485173366,157.3276829574384L201.67177919609884," \
              "152.45882785327547L199.6669565061494,145.0123435763204L199.95335974757074,138.99787550647207L202.24458567894152," \
              "135.84743985083725L202.5309889203629,133.55621391946644L202.5309889203629,133.84261716088778Z"
    path_to_coords = path_to_indices(svgpath)
    assert list(list(path_to_coords)[0]) == [223, 131]
    bool_inside = path_to_mask(svgpath, array.shape)
    x_inside = np.where(bool_inside == True)[1]
    y_inside = np.where(bool_inside == True)[0]
    assert np.min(x_inside) == 200
    assert np.max(x_inside) == 236
    assert np.min(y_inside) == 131
    assert np.max(y_inside) == 162
    assert bool_inside[131, 223]
    assert not bool_inside[130, 223]
    # Edit pixels inside and outside of the path to compute the statistics
    assert get_area_statistics_from_closed_path(array, svgpath) == (0.0, 0, 0, 0, 0, 0)
    array[130, 223] = 5000
    array[131, 237] = 5000
    assert get_area_statistics_from_closed_path(array, svgpath) == (0.0, 0, 0, 0, 0, 0)
    array[131, 223] = 5000
    mean, max, min, median, sd, total = get_area_statistics_from_closed_path(array, svgpath)
    assert mean > 0
    assert max == 5000.0
    assert min == 0.0
    assert total == max
    assert 170 < sd < 171
    array[150, 220] = 500
    mean_2, max_2, min_2, median_2, sd_2, total_2 = get_area_statistics_from_closed_path(array, svgpath)
    assert mean_2 > mean
    assert max_2 == 5000.0
    assert min_2 == 0.0
    assert 171 < sd_2 < 172
    array[152, 230] = -1.0
    assert total_2 > max_2
    mean_3, max_3, min_3, median_3, sd_3, total_3 = get_area_statistics_from_closed_path(array, svgpath)
    assert mean_2 > mean_3
    assert max_2 == 5000.0
    assert min_3 == -1.0
    assert 171 < sd_2 < 172
    assert get_bounding_box_for_svgpath(svgpath) == (200, 236, 131, 162)

def test_path_to_mask_over_boundary():
    # test that a path that goes over the border stops at the border
    border_annotation = "M598.1280487804879,225.87195121951223L507.57926829268297,235.01829268292684" \
                        "L476.48170731707324,245.07926829268294L458.1890243902439,299.0426829268293L" \
                        "445.3841463414634,360.3231707317073L447.2134146341464,387.7621951219512L" \
                        "458.1890243902439,410.6280487804878L479.22560975609764,423.4329268292683L" \
                        "537.7621951219513,449.95731707317077L557.8841463414635,449.0426829268293L" \
                        "599.5000000000001,437.1524390243903L599.0426829268293,206.66463414634148Z"

    mask = np.full((600, 600), 255).astype(np.uint8)
    bool_mask = path_to_mask(border_annotation, mask.shape)
    assert bool_mask[300, 500]
    assert bool_mask[300, 599]
    assert not bool_mask[366, 444]

def test_random_colour_selector():
    DEFAULT_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#00FAFF", "#FF00FF", "#FFFF00", "#FFFFFF"]

    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                   "Nuclear": np.array([1, 1, 1, 1]),
                                                   "Cytoplasm": np.array([2, 2, 2, 2])},
                   "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                   "Nuclear": np.array([4, 4, 4, 4]),
                                                   "Cytoplasm": np.array([5, 5, 5, 5])}
                   }
    blend_dict = create_new_blending_dict(upload_dict)
    assert blend_dict['Cytoplasm']['color'] == '#FFFFFF'
    blend_dict = select_random_colour_for_channel(blend_dict, "Nuclear", DEFAULT_COLOURS)
    assert DEFAULT_COLOURS[0] == blend_dict['Nuclear']['color']
    blend_dict = select_random_colour_for_channel(blend_dict, "Cytoplasm", DEFAULT_COLOURS)
    assert DEFAULT_COLOURS[1] == blend_dict['Cytoplasm']['color']
    blend_dict = select_random_colour_for_channel(blend_dict, "Nuclear", DEFAULT_COLOURS)
    assert DEFAULT_COLOURS[0] == blend_dict['Nuclear']['color']


def test_get_additive_image():
    layer_dict = {"channel_1": np.full((200, 200, 3), 3),
                  "channel_2": np.full((200, 200, 3), 6),
                  "channel_3": np.full((200, 200, 3), 9)}

    additive = get_additive_image(layer_dict, list(layer_dict.keys()))
    assert np.max(additive) == 18.0
    assert np.min(additive) == 18.0
    assert np.mean(additive) == 18.0

    layer_dict = {"channel_1": np.full((200, 200, 3), 1000),
                  "channel_2": np.full((200, 200, 3), 2000),
                  "channel_3": np.full((200, 200, 3), 3000)}

    additive = get_additive_image(layer_dict, list(layer_dict.keys()))
    assert np.max(additive) == 6000.0
    assert np.min(additive) == 6000.0
    assert np.mean(additive) == 6000.0

    assert get_additive_image(layer_dict, []) is None

def test_retrieval_first_roi_dict_image():
    layer_dict = {"channel_1": np.full((200, 200, 3), 1000),
                  "channel_2": np.full((200, 200, 3), 2000),
                  "channel_3": np.full((200, 200, 3), 3000)}
    first_array = get_first_image_from_roi_dictionary(layer_dict)
    assert first_array.shape == (200, 200, 3)
    assert np.mean(first_array) == 1000

def test_apply_filter_to_array(get_current_dir):
    greyscale = np.array(Image.open(os.path.join(get_current_dir, "for_recolour.tiff")))
    median_filter_3 = apply_filter_to_array(greyscale, ['apply filter'], "median", 3, 1)
    assert not np.array_equal(greyscale, median_filter_3)

    median_filter_1 = apply_filter_to_array(greyscale, True, "median", 1, 1)
    assert np.array_equal(greyscale, median_filter_1)

    gaussian_filter_5 = apply_filter_to_array(greyscale, True, "gaussian", 5, 1)
    assert not np.array_equal(greyscale, gaussian_filter_5)

    gaussian_filter_1 = apply_filter_to_array(greyscale, True, "gaussian", 1, 1)
    assert np.array_equal(greyscale, gaussian_filter_1)

    median_negative = apply_filter_to_array(greyscale, True, "median", -1, 0)
    assert np.array_equal(greyscale, median_negative)

    gaussian_negative = apply_filter_to_array(greyscale, True, "gaussian", -1, 0)
    assert np.array_equal(greyscale, gaussian_negative)

    even_value = apply_filter_to_array(greyscale, True, "median", 4, 0)
    assert np.array_equal(greyscale, even_value)

    with pytest.raises(TypeError):
        apply_filter_to_array(greyscale, True, "fake_filter", 3, 0)

    median_high = apply_filter_to_array(greyscale, True, "median", 99, 0)
    assert np.array_equal(greyscale, median_high)

    no_filter = apply_filter_to_array(greyscale, [], "median", 1, 1)
    assert np.array_equal(greyscale, no_filter)

    no_filter = apply_filter_to_array(greyscale, False, "median", 1, 1)
    assert np.array_equal(greyscale, no_filter)


def test_filter_bool_eval():
    blend_dict = {"channel_1": {"color": "#FFFFFF", "x_lower_bound": None,
                "x_upper_bound": None, "filter_type": None, "filter_val": None, "filter_sigma": None}}
    assert no_filter_chosen(blend_dict, "channel_1", [])
    blend_dict = {"channel_1": {"color": "#FFFFFF", "x_lower_bound": None,
                "x_upper_bound": None, "filter_type": "median", "filter_val": None, "filter_sigma": None}}
    assert not no_filter_chosen(blend_dict, "channel_1", [])

    blend_dict = {"channel_1": {"color": "#FFFFFF", "x_lower_bound": None,
                                "x_upper_bound": None, "filter_type": "gaussian", "filter_val": 5,
                                "filter_sigma": 0.5}}
    # assert no match if the filter is not currently applied
    assert channel_filter_matches(blend_dict, "channel_1", [' apply filter'], "gaussian", 5, 0.5)
    assert not channel_filter_matches(blend_dict, "channel_1", [], "gaussian", 5, 0.5)

def test_ag_grid_cell_styling():
    blend_dict = {"channel_1": {"color": "#FFFFFF"}, "channel_2": {"color": "#E22424"},
                  "channel_3": {"color": "#CCFFE5"}}
    aliases = {"channel_1": "ch1", "channel_2": "ch2", "channel_3": "ch3"}
    cell_styling = ag_grid_cell_styling_conditions(blend_dict, list(blend_dict.keys()) + ["channel_4"], "roi_1", aliases)
    assert cell_styling == [{'condition': "params.value == 'ch1'", 'style': {'color': 'black'}},
                            {'condition': "params.value == 'ch2'", 'style': {'color': '#E22424'}},
                            {'condition': "params.value == 'ch3'", 'style': {'color': '#CCFFE5'}}]

def test_extract_zoom_bounds():
    bounds = {'xaxis.range[0]': 597.512350562311, 'xaxis.range[1]': 767.7478344332787,
              'yaxis.range[0]': 419.1645161290323, 'yaxis.range[1]': 309.7838709677419}
    x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(bounds)
    assert x_low == bounds['xaxis.range[0]']
    assert y_high == bounds['yaxis.range[0]']
    # as int
    x_low, x_high, y_low, y_high = high_low_values_from_zoom_layout(bounds, cast_type=int)
    assert x_low == 597

def test_marker_correlation_metrics(get_current_dir):
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff"))).astype(np.float32)

    # case 1: no overlap between target and baseline in mask
    image_dict = {"roi_1": {"target": np.where(mask == 1.0, 1000, 0).astype(np.float32),
                            "baseline": np.where(mask > 1, 1000, 0).astype(np.float32)}}
    blend_dict = {"target": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
     'filter_type': 'median', 'filter_val': 3, 'filter_sigma': 1},
                  "baseline": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
     'filter_type': None, 'filter_val': 3, 'filter_sigma': 1}}
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", "baseline",
                                        mask=mask, blend_dict=blend_dict).get_correlation_statistics()
    # assert that all the proportion is inside the mask, but there is no overlap with the baseline
    assert float(proportion_target) == 1.0
    assert overlap == 0.0
    # assert that the proportion of base line is entirely in the mask
    assert proportion_baseline == 1.0

    # repeat above with bounds
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", "baseline",
                            mask=mask,
        blend_dict=None, bounds={'xaxis.range[0]': 10, 'xaxis.range[1]': 100,
                                       'yaxis.range[1]': 110, 'yaxis.range[0]': 0}).get_correlation_statistics()
    assert np.isnan(proportion_target)

    # use bounds that cause an index error
    # on index error, just use the default bounds
    assert all([elem is not None for elem in MarkerCorrelation(image_dict, "roi_1", "target", "baseline", mask=mask,
        blend_dict=None, bounds={'xaxis.range[0]': -10, 'xaxis.range[1]': 100000,
        'yaxis.range[1]': -110, 'yaxis.range[0]': 0}).get_correlation_statistics()])

    # have complete overlap between target and baseline in mask
    image_dict = {"roi_1": {"target": np.where(mask > 0, 1000, 0).astype(np.float32),
                            "baseline": np.where(mask > 0, 1000, 0).astype(np.float32)}}
    blend_dict = {"target": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
                             'filter_type': None, 'filter_val': 3, 'filter_sigma': 1},
                  "baseline": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
                               'filter_type': None, 'filter_val': 3, 'filter_sigma': 1}}
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", "baseline",
                                        mask=mask, blend_dict=blend_dict).get_correlation_statistics()
    assert proportion_target == overlap == proportion_baseline == 1.0

    # when using a filter, some signal will spill outside of the mask boundaries
    blend_dict = {"target": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
                             'filter_type': 'gaussian', 'filter_val': 5, 'filter_sigma': 1},
                  "baseline": {'color': '#BE4115', 'x_lower_bound': 0, 'x_upper_bound': None,
                               'filter_type': None, 'filter_val': 3, 'filter_sigma': 1}}
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", "baseline",
                                        mask=mask, blend_dict=blend_dict).get_correlation_statistics()

    assert 0.88 < proportion_target < overlap == proportion_baseline == 1.0

    # no baseline
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", None, mask=mask,
                                            blend_dict=blend_dict).get_correlation_statistics()
    assert overlap is None
    assert proportion_baseline is None

    # if there is no target, all is None
    assert MarkerCorrelation(image_dict, "roi_1", None, "baseline", mask=mask,
            blend_dict=blend_dict).get_correlation_statistics() == (None, None, None, None)

    # mask that is not the same shape, so there is correlation but nothing else
    proportion_target, overlap, proportion_baseline, pearson = MarkerCorrelation(image_dict, "roi_1", "target", "baseline", mask=None,
                             blend_dict=blend_dict).get_correlation_statistics()
    assert (proportion_target, overlap, proportion_baseline) == (None, None, None)
    assert 0.96 < pearson < 0.97

    blend_dict = {"target": {'color': '#BE4115', 'x_lower_bound': 10, 'x_upper_bound': None,
                             'filter_type': 'gaussian', 'filter_val': 5, 'filter_sigma': 1},
                  "baseline": {'color': '#BE4115', 'x_lower_bound': 10, 'x_upper_bound': None,
                               'filter_type': None, 'filter_val': 3, 'filter_sigma': 1}}

    # assert that the correlation changes when the bounds change
    mask_bad_shape = np.zeros((999, 999))
    proportion_target, overlap, proportion_baseline, pearson_2 = MarkerCorrelation(image_dict, "roi_1",
                    "target", "baseline", mask=mask_bad_shape, blend_dict=blend_dict).get_correlation_statistics()
    assert (proportion_target, overlap, proportion_baseline) == (None, None, None)
    assert pearson_2 != pearson

def test_saving_blend():
    saved = add_saved_blend(None, "infiltration", ["chan_1", "chan_4", "chan_5"])
    assert saved == {'infiltration': ['chan_1', 'chan_4', 'chan_5']}
    saved = add_saved_blend(saved, "immune", ["CD8", "CD3"])
    assert saved == {'immune': ['CD8', 'CD3'], 'infiltration': ['chan_1', 'chan_4', 'chan_5']}
    saved = add_saved_blend(saved, None, ["chan_11", "chan_12"])
    assert saved == {'immune': ['CD8', 'CD3'], 'infiltration': ['chan_1', 'chan_4', 'chan_5']}
    saved = add_saved_blend(saved, "empty")
    assert saved == {'immune': ['CD8', 'CD3'], 'infiltration': ['chan_1', 'chan_4', 'chan_5']}
