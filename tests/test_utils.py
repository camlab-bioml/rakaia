import pytest
from ccramic.app.utils import *
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
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            if original_pixels[i, j] < 51:
                assert new_pixels[i, j] == 0
            else:
                assert new_pixels[i, j] >= 51

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
    for i in range(greyscale_image.height):
        for j in range(greyscale_image.width):
            if original_pixels[i, j] < 1:
                assert new_pixels[i, j] == 0
            else:
                assert 15 <= new_pixels[i, j]

    assert np.max(greyscale) >= np.max(filtered_1)


def test_generate_histogram(get_current_dir):
    greyscale_image = Image.open(os.path.join(get_current_dir, "for_recolour.tiff"))
    greyscale = np.array(greyscale_image)
    histogram = pixel_hist_from_array(greyscale)
    assert isinstance(histogram, plotly.graph_objs._figure.Figure)
    assert histogram["data"] is not None
    assert histogram["layout"] is not None
    values = histogram["data"][0]['x']
    assert len(values) == 360001
    assert max(values) == np.max(greyscale)
