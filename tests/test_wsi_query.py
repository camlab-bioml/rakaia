import os
import numpy as np
import io
from unittest.mock import Mock, patch
from rakaia.register.query import (
    wsi_crop,
    serialize_crop,
    tcga_uni_request,
    TCGA_UNI_COL_DEFS,
    format_col_ag_groupings)

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
    assert serialize_crop(None) is None

def test_toggle_tcga_col_defs():
    assert format_col_ag_groupings() == TCGA_UNI_COL_DEFS
    groupings_off = format_col_ag_groupings(False)
    for col in groupings_off:
        if 'rowGroup' in col:
            assert not col['rowGroup']
    assert not groupings_off == TCGA_UNI_COL_DEFS

@patch("rakaia.register.query.requests.post")
def test_tcga_post(mock_post):
    expected = [
        {"slide": "TCGA-01", "score": 0.5, "url": "url-1"},
        {"slide": "TCGA-02", "score": 0.35, "url": "url-2"}]

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = expected
    mock_post.return_value = mock_response

    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    result = tcga_uni_request(crop)
    assert result == expected
    mock_post.assert_called_once()

    assert tcga_uni_request(None) is None
    mock_post.assert_called_once()
