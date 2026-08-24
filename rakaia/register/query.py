"""
Module related to functions and classes for processing WSI patches and enabling API queries
"""
import io
import copy
from pathlib import Path
from http.client import HTTPException
from typing import Union
from pathlib import Path
import textwrap
import pandas as pd
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go

# Default column definitions for the TCGA UNI search results shown in dash ag grid
TCGA_UNI_COL_DEFS = [{"field": "tissue", "rowGroup": True, "hide": True}, {"field": "slide", "rowGroup": True, "hide": True},
                    {"field": "x"}, {"field": "y"},
                    {"field": "url", "cellRenderer": "LinkRenderer"}, {"field": "similarity"},
                    {"field": "open", "headerName": "Open patch in slide viewer", "cellRenderer": "OpenSlideButton",
                    "cellRendererParams": {"className": "open-slide-btn"},
                    # static label shown on every button in this column
                    "valueGetter": {"function": "'Open'"}, "sortable": False, "filter": False}]

def wsi_crop(image: Union[Path, str, np.ndarray, None],
             bounds: Union[list, None]=None,
             return_sampled: bool=True,
             patch_out_size: int=224):
    """
    Generate a crop of a WSI image processed through pyvips. Assumes tha the bounds array is in the
    format `[x0, x1, y0, y1]`.
    If `return_subsample` is used, specify the size (i.e. 224 works for UNI patch embeddings)
    """
    import pyvips
    try:
        x0, x1, y0, y1 = bounds
        crop = pyvips.Image.new_from_file(image, access="sequential") if not \
            isinstance(image, np.ndarray) else pyvips.Image.new_from_array(image, interpretation='rgb')
        crop = crop.crop(x0, y0, x1 - x0, y1 - y0).numpy().astype(np.uint8)
        # drop alpha channel if present, often from svs
        if (len(crop.shape) == 3) and crop.shape[2] == 4: crop = crop[:, :, :3]
        # TODO: how should the aspect ratio be handled? Here we make a square subsample patch
        # square sub-sampled patches appear to work well for UNI2, but may not for Prism2
        return np.array(Image.fromarray(crop).resize((patch_out_size, patch_out_size),
             resample=Image.Resampling.LANCZOS)) if (return_sampled and patch_out_size > 0) else crop
    except (pyvips.Error, TypeError, KeyError): pass
    return None

def serialize_crop(crop: Union[np.array, np.ndarray, None]=None):
    """
    Serialize the WSI crop into compressed bytes for a POST request
    """
    if crop is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=np.stack(crop))
        return buffer.getvalue()
    return None

def tcga_resp_to_table(resp: Union[dict, None]=None):
    """
    Format the TCGA UNI POST response into a table (record-oriented) for viewing
    """
    if resp is not None and all(elem in resp.keys() for elem in ('hits', 'url')):
        hits_frame = pd.DataFrame(resp['hits'])
        hits_frame['url'] = "NA"
        if resp['url'] and isinstance(resp['url'], dict):
            hits_frame['url'] = hits_frame['slide'].map(resp['url'])
        return hits_frame.to_dict(orient="records")
    return None

def set_query_host(api_host: str="localhost",
                   api_port: int=6000):
    """
    Set the host and port for hist2query. Accepts localhost + port or a URL
    """
    if str(api_host).startswith("http") or str(api_host).startswith("https"): return api_host
    return f"http://{api_host}:{api_port}"

def tcga_uni_request(crop: Union[np.ndarray, np.array, None]=None,
                            api_host: str="localhost",
                            api_port: int=7000,
                            k_search: int=10,
                            return_url: bool=True,
                            endpoint: str="search",
                            return_processed: bool=True):
    """
    Format the TCGA UNI POST request to send to hist2query
    """
    if crop is not None:
        response = requests.post(f"{set_query_host(api_host, api_port)}/{endpoint}",
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"k": k_search, "url": return_url}, timeout=300)
        resp = handle_request_error(response)
        return tcga_resp_to_table(resp) if return_processed else resp
    return None

def handle_request_error(response: requests.request):
    """
    Handle the HTTP error and exceptions from hist2query
    """
    try:
        response.raise_for_status()
    except (requests.exceptions.HTTPError, HTTPException) as e:
        raise HTTPException(response.status_code, e.response.json().get("detail", "Error"))
    return response.json()

def prism2_chat_request(crop: Union[np.ndarray, np.array, None]=None,
                            api_host: str="localhost",
                            api_port: int=7000,
                            endpoint: str="chat",
                            question: str="What type of tissue is this?"):
    if crop is not None:
        response = requests.post(f"{set_query_host(api_host, api_port)}/{endpoint}",
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"question": question, "raw_scores_binary": False,
                                       "max_token_response": 100}, timeout=300)
        resp = handle_request_error(response)
        return resp['response'][0] if isinstance(resp['response'], list) else str(resp['response'])
    return None

def format_col_ag_groupings(use_grouping: bool=True):
    """
    format the col groupings
    """
    new_col_defs = copy.deepcopy(TCGA_UNI_COL_DEFS)
    for col in new_col_defs:
        if "rowGroup" in col:
            col["rowGroup"] = use_grouping
            col["hide"] = use_grouping
    return new_col_defs

def tile_dimension_labels(max_dim: int=10):
    """
    Set the `dcc.Dropdown` options and labels for the hist2query tile number downsample
    """
    return [{"label": f"{val}x{val}", "value": val} for val in range(1, int(max_dim + 1))]

def hist2query_pie_chart(query_results: Union[list, dict, None],
                    category: str="tissue"):
    """
    Generate a pie chart of the hist2query results by tissue or project
    """
    if None not in (query_results, category):
        query_results_plot = pd.DataFrame(query_results)
        grouping_col = category if category in query_results_plot.columns else "project"
        query_results_plot[grouping_col] = query_results_plot[grouping_col].apply(
            lambda x: "<br>".join(textwrap.wrap(x, width=25)))
        fig = go.Figure(px.pie(query_results_plot[grouping_col]
                               .value_counts(dropna=False)
                               .rename_axis("Tissue Type")
                               .reset_index(name="Count"),
                               names="Tissue Type",
                               values="Count",
                               title=f"Query {str(grouping_col)} distribution"))
        fig.update_layout(autosize=True, margin=dict(l=0, r=50, t=50, b=0),
                          legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1))
        return fig.to_dict()
    return None

_GDC_SlIDE_TEMPLATE = (Path(__file__).parent / "../templates" / "gdc.html").read_text()

def gdc_slide_iframe(file_id: str, x: float, y: float, n: float = 1250) -> str:
    """
    Generate a GDC slide OSD viewer for a specific GDC-hosted slide with a patch highlighted by coordinates
    and a bounding box. Compatible with hist2query patch results
    """
    return (_GDC_SlIDE_TEMPLATE
        .replace("__FILE_ID__", str(file_id))
        .replace("__X__", repr(float(x)))
        .replace("__Y__", repr(float(y)))
        .replace("__N__", repr(float(n))))
