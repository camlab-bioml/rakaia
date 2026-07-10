import os
import numpy as np
import io
from rakaia.register.query import wsi_crop, serialize_crop

def test_wsi_roi_crop(get_current_dir):
    wsi = os.path.join(get_current_dir, 'for_recolour.tiff')
    bounds = [0, 300, 0, 400]
    crop = wsi_crop(wsi, bounds, False)
    assert crop.shape == (400, 300)
    crop_subsample = wsi_crop(wsi, bounds)
    assert crop_subsample.shape == (224, 224)

    assert wsi_crop(os.path.join(get_current_dir, 'query_from_text.txt'), bounds) is None
    assert wsi_crop(None, None) is None

    bytes_buffer = serialize_crop(crop_subsample)
    assert isinstance(bytes_buffer, bytes)
    reconverted = np.load(io.BytesIO(bytes_buffer))['data']
    np.testing.assert_array_equal(crop_subsample, reconverted)
