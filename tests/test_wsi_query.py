import os
from http.client import HTTPException
import numpy as np
import io
from unittest.mock import Mock, patch
import pandas as pd
import pytest
import requests
from rakaia.register.query import (
    wsi_crop,
    serialize_crop,
    tcga_uni_request,
    TCGA_UNI_COL_DEFS,
    format_col_ag_groupings,
    tcga_resp_to_table,
    hist2query_pie_chart,
    prism2_chat_request,
    gdc_slide_iframe,
    tile_dimension_labels,
    hist2query_clinical_bar_plot,
    hist2query_tissue_list, set_tcga_metadata_options)

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
    expected = {'hits': [
        {"slide": "TCGA-01", "score": 0.5},
        {"slide": "TCGA-02", "score": 0.35}],
    'url': {"TCGA-01": 'url_1', "TCGA-02": 'url_2'}}

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = expected
    mock_post.return_value = mock_response

    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    assert tcga_resp_to_table(None) is None
    assert tcga_resp_to_table({'fake': None}) is None

    result = tcga_uni_request(crop)
    assert len(result) == 2
    assert all(elem['url'] != "NA" for elem in result)
    mock_post.assert_called_once()

    assert tcga_uni_request(None) is None
    mock_post.assert_called_once()

@patch("rakaia.register.query.requests.post")
def test_prism2_chat_post(mock_post):
    expected = {'response': ['This is cancerous tissue.']}

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = expected
    mock_post.return_value = mock_response

    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    result = prism2_chat_request(crop)
    assert str(result) == 'This is cancerous tissue.'

    assert tcga_uni_request(None) is None
    mock_post.assert_called_once()

@patch("rakaia.register.query.requests.post")
def test_prism2_chat_post_exception(mock_post):

    mock_response = Mock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "503 Service Unavailable")

    error = requests.exceptions.HTTPError("503 Service Unavailable")
    error.response = mock_response

    mock_response.raise_for_status.side_effect = error
    mock_post.return_value = mock_response

    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(HTTPException):
        prism2_chat_request(crop)

    assert prism2_chat_request(None) is None
    mock_post.assert_called_once()

def test_set_tile_dim_labels():
    options = tile_dimension_labels()
    assert len(options) == 10
    assert options[0]['value'] == 1
    assert options[-1]['value'] == 10

def test_hist2query_results_chart():
    hits = [
        {"tissue": "kidney", "slide": "TCGA-01", "score": 0.5},
        {"tissue": "kidney", "slide": "TCGA-02", "score": 0.35},
        {"tissue": "breast", "slide": "TCGA-03", "score": 0.7},
        {"tissue": "breast", "slide": "TCGA-04", "score": 0.6}]
    dist_tissue = hist2query_pie_chart(hits)
    assert 'Tissue Type' in dist_tissue['data'][0]['hovertemplate']
    assert 'kidney' in dist_tissue['data'][0]['labels']
    assert dist_tissue['data'][0]['type'] == 'pie'
    assert hist2query_pie_chart(None) is None


def test_gdc_slide_iframe():
    gdc_html_template = gdc_slide_iframe("new_slide", "new_slide", 1000, 1000, 2000)
    assert gdc_html_template.startswith("<!DOCTYPE html>")
    assert 'const FILE_ID = "new_slide";' in gdc_html_template
    assert 'const URL = "new_slide";' in gdc_html_template

def test_tcga_metadata_barplot():

    assert not 'bcr_patient_barcode' in set_tcga_metadata_options()
    assert 'gender' in set_tcga_metadata_options()

    results = [{'project': 'TCGA-KIRC', 'tissue': 'Kidney renal clear cell carcinoma',
                'slide': 'TCGA-T7-A92I-01Z-00-DX2.h5', 'x': 23040, 'y': 77312,
                'similarity': 0.6102864742279053,
                'url': 'https://portal.gdc.cancer.gov/files/5de05155-274d-4310-83bd-4f24e00af6ee'},
               {'project': 'TCGA-KIRC', 'tissue': 'Kidney renal clear cell carcinoma', 'slide':
                   'TCGA-T7-A92I-01Z-00-DX1.h5', 'x': 58368, 'y': 57344, 'similarity': 0.6087409853935242,
                'url': 'https://portal.gdc.cancer.gov/files/430a1f3c-5c08-44cf-8dd1-ff1d954175a5'},
               {'project': 'TCGA-KIRC', 'tissue': 'Kidney renal clear cell carcinoma', 'slide':
                   'TCGA-T7-A92I-01Z-00-DX3.h5', 'x': 29440, 'y': 82944, 'similarity': 0.6076342463493347,
                'url': 'https://portal.gdc.cancer.gov/files/3eabeeea-1f5b-4e8c-b343-200fcdd19a5b'},
               {'project': 'TCGA-KIRC', 'tissue': 'Kidney renal clear cell carcinoma', 'slide':
                   'TCGA-T7-A92I-01Z-00-DX1.h5', 'x': 38912, 'y': 41472, 'similarity': 0.6048403382301331,
                'url': 'https://portal.gdc.cancer.gov/files/430a1f3c-5c08-44cf-8dd1-ff1d954175a5'},
               {'project': 'TCGA-KIRC', 'tissue': 'Kidney renal clear cell carcinoma', 'slide':
                   'TCGA-T7-A92I-01Z-00-DX2.h5', 'x': 25856, 'y': 80896, 'similarity': 0.6034709215164185,
                'url': 'https://portal.gdc.cancer.gov/files/5de05155-274d-4310-83bd-4f24e00af6ee'},
               {'project': 'TCGA-KIRP', 'tissue': 'Kidney renal papillary cell carcinoma', 'slide':
                   'TCGA-5P-A9JY-01Z-00-DX1.h5', 'x': 57856, 'y': 129536, 'similarity': 0.6030967831611633,
                'url': 'https://portal.gdc.cancer.gov/files/98463c73-ebcf-469b-9dad-71c1453c0386'}]

    assert len(hist2query_tissue_list(results)) == 2
    assert len(hist2query_tissue_list(None)) == 0

    bar_meta, prop = hist2query_clinical_bar_plot(pd.DataFrame(results), 'histological_type')
    assert len(bar_meta['data']) == 2
    assert '(2 patients, 6/6 query patches)' in bar_meta['layout']['title']['text']
    assert len(prop) == 2
    assert pd.DataFrame(prop)['Proportion'].to_list() == [0.5, 0.5]

    bar_meta_sub, prop = hist2query_clinical_bar_plot(results, 'histological_type',
                                             subset_tissue_groups=['Kidney renal papillary cell carcinoma'])
    assert len(bar_meta_sub['data']) == 1
    assert len(prop) == 1
    assert '(1 patients, 1/6 query patches)' in bar_meta_sub['layout']['title']['text']

    assert hist2query_clinical_bar_plot(results, None) == (None, None)
