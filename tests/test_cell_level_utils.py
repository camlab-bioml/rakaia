from ccramic.app.utils.cell_level_utils import *
import os
from PIL import Image

def test_basic_mask_boundary_converter(get_current_dir):
    fake_mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    mask_copy = fake_mask.copy()
    assert np.array_equal(fake_mask, mask_copy)
    mask_with_boundary = convert_mask_to_cell_boundary(fake_mask)
    assert not np.array_equal(fake_mask, mask_with_boundary)
    assert np.max(mask_with_boundary) == 255
    assert not np.max(fake_mask) == 255
    reconverted_back = np.array(Image.fromarray(fake_mask).convert('RGB').convert('L'))
    assert np.max(reconverted_back) == 255
    # assert that the boundary conversion makes the overall mean less because white pixels on the interior of
    # the cell are converted to black
    # need to convert the mask to RGB then back to greyscale to ensure that the mask max is 255 for standard intensity
    # comparison
    mask_from_reconverted = convert_mask_to_cell_boundary(reconverted_back)
    assert np.mean(reconverted_back) > np.mean(mask_from_reconverted)
    # assert that a pixel inside the cell boundary is essentially invisible
    assert reconverted_back[628, 491] == 255
    assert mask_from_reconverted[628, 491] <= 3
