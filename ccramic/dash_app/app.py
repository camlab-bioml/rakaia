import time

import anndata
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from PIL import Image, ImageSequence

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import os
from io import BytesIO
import base64
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import numpy as np
import flask
import tempfile
import dash_uploader as du
from dash_canvas import DashCanvas
import uuid
from dash import callback_context, no_update
import plotly.express as px
import io
from flask_caching import Cache
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash_daq as daq

app = DashProxy(transforms=[ServersideOutputTransform()])
app.title = "ccramic"

try:
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    })
except (ModuleNotFoundError, RuntimeError) as no_redis:
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'cache-directory'
    })

with tempfile.TemporaryDirectory() as tmpdirname:
    du.configure_upload(app, tmpdirname)


def convert_image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@du.callback(ServersideOutput('uploaded_dict', 'data'),
             id='upload-image')
@cache.memoize(timeout=60)
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    if filenames:
        upload_dict = {}
        upload_index = 0
        for uploaded in filenames:
            layer_index = 0
            file_designation = str(uuid.uuid4()) + str(upload_index)
            image = Image.open(uploaded)
            image.load()
            for i, page in enumerate(ImageSequence.Iterator(image)):
                layer_designation = "_layer_" + str(layer_index)
                page = page.convert('RGB')
                upload_dict[file_designation + layer_designation] = page
                # upload_dict[file_designation + layer_designation] = convert_image_to_bytes(page)
                layer_index += 1
        if upload_dict:
            return upload_dict
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@du.callback(ServersideOutput('anndata', 'data'),
             id='upload-quantification')
@cache.memoize(timeout=60)
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    anndata_files = {}
    if filenames:
        # return anndata.read_h5ad(filenames[0])
        index = 0
        for data_file in filenames:
            anndata_dict = {}
            data = anndata.read_h5ad(data_file)
            anndata_dict["file_path"] = str(data_file)
            anndata_dict["observations"] = data.X
            anndata_dict["metadata"] = data.obs
            anndata_dict["full_obj"] = data
            for sub_assay in data.obsm_keys():
                if "assays" not in anndata_dict.keys():
                    anndata_dict["assays"] = {sub_assay: data.obsm[sub_assay]}
                else:
                    anndata_dict["assays"][sub_assay] = data.obsm[sub_assay]
            # anndata_files["anndata_" + str(index)] = anndata_dict
            anndata_files = anndata_dict
    if anndata_files is not None and len(anndata_files) > 0:
        return anndata_files
    else:
        raise PreventUpdate


@app.callback(Output('dimension-reduction_options', 'options'),
              Input('anndata', 'data'))
def create_anndata_dimension_options(anndata_dict):
    if anndata_dict and "assays" in anndata_dict.keys():
        print("updating anndata")
        return [{'label': i, 'value': i} for i in anndata_dict["assays"].keys()]
    else:
        raise PreventUpdate


@app.callback(Output('metadata_options', 'options'),
              Input('anndata', 'data'))
def create_anndata_dimension_options(anndata_dict):
    if anndata_dict and "metadata" in anndata_dict.keys():
        return [{'label': i, 'value': i} for i in anndata_dict["metadata"].columns]
    else:
        raise PreventUpdate


@app.callback(Output('image_layers', 'options'),
              Input('uploaded_dict', 'data'))
def create_dropdown_options(image_dict):
    if image_dict:
        return [{'label': i, 'value': i} for i in image_dict.keys()]
    else:
        raise PreventUpdate


def read_back_base64_to_image(string):
    image_back = base64.b64decode(string)
    return Image.open(io.BytesIO(image_back))


@app.callback(Output('annotation_canvas', 'figure'),
              Input('image_layers', 'value'),
              Input('uploaded_dict', 'data'),
              Input("annotation-color-picker", "value"))
def render_image_on_canvas(image_str, image_dict, annotation_color):
    if image_str is not None and image_str in image_dict.keys():
        fig = px.imshow(image_dict[image_str], aspect='auto')
        fig.update_layout(
            newshape=dict(fillcolor=annotation_color["hex"], line=dict(color=annotation_color["hex"])))
        return fig
    else:
        raise PreventUpdate


@app.callback(Output('umap-plot', 'figure'),
              Input('anndata', 'data'),
              Input('metadata_options', 'value'),
              Input('dimension-reduction_options', 'value'))
def render_umap_plot(anndata, metadata_selection, assay_selection):
    if anndata and "assays" in anndata.keys() and metadata_selection and assay_selection:
        data = anndata["full_obj"]
        return px.scatter(data.obsm[assay_selection], x=0, y=1, color=data.obs[metadata_selection],
                          labels={'color': metadata_selection})
    else:
        raise PreventUpdate


app.layout = html.Div([
    html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
    dcc.Tabs([
        dcc.Tab(label='Image Annotation', children=[
            du.Upload(
                id='upload-image',
                max_file_size=1800,  # 1800 Mb
                filetypes=['png', 'tif', 'tiff', 'csv'],
                upload_id="upload-image",
            ),
            dcc.Dropdown(id='image_layers'),
            daq.ColorPicker(id="annotation-color-picker", label="Color Picker", value=dict(hex="#119DFF")),
            html.H3("Annotate your tif file"), dcc.Graph(config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape"]}, id='annotation_canvas', style={'width': '150vh', 'height': '150vh'}),
        ]),
        dcc.Tab(label='Quantification/Clustering', children=[
            du.Upload(
                id='upload-quantification',
                max_file_size=1800,  # 1800 Mb
                filetypes=['h5ad', 'h5'],
                upload_id="upload-quantification"),
            dcc.Dropdown(id='dimension-reduction_options'),
            dcc.Dropdown(id='metadata_options'),
            dcc.Graph(id='umap-plot', style={'width': '150vh', 'height': '150vh'})
        ])
    ]),
    dcc.Loading(dcc.Store(id="uploaded_dict"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="anndata"), fullscreen=True, type="dot"),
])
