import pytest
from ccramic.app.utils import *
from ccramic.app.parsers import *
import os
from PIL import Image, ImageColor
import tifffile
import plotly


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


def test_generate_histogram(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    histogram, array_max = pixel_hist_from_array(greyscale)
    assert isinstance(histogram, plotly.graph_objs._figure.Figure)
    assert histogram["data"] is not None
    assert histogram["layout"] is not None
    values = histogram["data"][0]['x']
    assert len(values) == 360001
    assert array_max == int(np.max(greyscale))


def test_basic_blend_dict_params():

    upload_dict = {"experiment0": {"slide0": {"acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                       "Nuclear": np.array([1, 1, 1, 1]),
                                                       "Cytoplasm": np.array([2, 2, 2, 2])},
                                              "acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                       "Nuclear": np.array([4, 4, 4, 4]),
                                                       "Cytoplasm": np.array([5, 5, 5, 5])}
                                              }}}

    blend_dict = create_new_blending_dict(upload_dict)
    assert len(upload_dict.keys()) == 1
    for exp in upload_dict.keys():
        assert len(upload_dict[exp].keys()) == 1
        assert len(blend_dict[exp].keys()) == 1
        for slide in upload_dict[exp].keys():
            assert len(upload_dict[exp][slide].keys()) == 2
            assert len(blend_dict[exp][slide].keys()) == 2
            for acq in upload_dict[exp][slide].keys():
                assert len(upload_dict[exp][slide][acq].keys()) == 3
                assert len(blend_dict[exp][slide][acq].keys()) == 3

    blend_dict["experiment0"]['slide0']["acq0"]["DNA"] = {'color': '#BE4115',
                                                          'x_lower_bound': 200,
                                                          'x_upper_bound': 1000,
                                                          'y_ceiling': 12500,
                                                          'filter_type': 'gaussian',
                                                          'filter_val': 5}

    blend_dict["experiment0"]['slide0']["acq0"]["Nuclear"] = {'color': '#15BEB0',
                                                          'x_lower_bound': 0.25,
                                                          'x_upper_bound': 55,
                                                          'y_ceiling': 67000,
                                                          'filter_type': 'median',
                                                          'filter_val': 3}

    blend_dict["experiment0"]['slide0']["acq0"]["Cytoplasm"] = {'color': '#BA15BE',
                                                          'x_lower_bound': -0.4,
                                                          'x_upper_bound': 14,
                                                          'y_ceiling': 900,
                                                          'filter_type': 'median',
                                                          'filter_val': 7}

    # check that the default values are either hex white or None
    possibilities = ['#FFFFFF', None]

    # assert that the firdst ROI has non default params
    for channel in blend_dict["experiment0"]['slide0']["acq0"].keys():
        for blend_param in blend_dict["experiment0"]['slide0']["acq0"][channel].values():
            assert blend_param not in possibilities

    # assert that the default second ROI has default params
    for channel in blend_dict["experiment0"]['slide0']["acq1"].keys():
        for blend_param in blend_dict["experiment0"]['slide0']["acq1"][channel].values():
            assert blend_param in possibilities

    blend_dict = copy_values_within_nested_dict(blend_dict, "experiment0+++slide0+++acq0",
                                                "experiment0+++slide0+++acq1")

    # assert that the default parameters in the second ROi are overwritten
    for channel in blend_dict["experiment0"]['slide0']["acq1"].keys():
        for blend_param in blend_dict["experiment0"]['slide0']["acq1"][channel].values():
            assert blend_param not in possibilities


def test_calculate_percentile_intensity(get_current_dir):
    array = np.array(Image.open(os.path.join(get_current_dir, "for_recolour.tiff")))
    default_val = get_default_channel_upper_bound_by_percentile(array=array)
    assert default_val == 70.67082221984863
    lower_val = get_default_channel_upper_bound_by_percentile(array=array, percentile=50)
    assert lower_val == 10.202707290649414
    assert default_val > lower_val
    assert get_default_channel_upper_bound_by_percentile(array=array, percentile=100) == float(np.max(array))
