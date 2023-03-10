import time

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


app = Dash(__name__)
app.title = "ccramic"

with tempfile.TemporaryDirectory() as tmpdirname:
    du.configure_upload(app, tmpdirname)

UPLOADED_DICT = {}


def initialize_uploaded_dict():
    UPLOADED_DICT.clear()


def append_to_uploaded_dict(key, value):
    UPLOADED_DICT[key] = value


@du.callback(Output('image_layers', 'options'),
              id='upload-image')
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    if filenames:
        initialize_uploaded_dict()
        upload_dict = []
        upload_index = 0
        for uploaded in filenames:
            layer_index = 0
            file_designation = str(uuid.uuid4()) + str(upload_index)
            image = Image.open(uploaded)
            image.load()
            for i, page in enumerate(ImageSequence.Iterator(image)):
                layer_designation = "_layer_" + str(layer_index)
                append_to_uploaded_dict(file_designation + layer_designation, page.convert('RGB'))
                upload_dict.append(file_designation + layer_designation)
                layer_index += 1
        if upload_dict:
            return [{'label': i, 'value': i} for i in upload_dict]
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(Output('output_cell_image', 'children'),
              Input('image_layers', 'value'))
def render_image(image):
    if image is not None and image in UPLOADED_DICT.keys():
        return html.Div([
        html.Img(src=UPLOADED_DICT[image])
            ])
    else:
        raise PreventUpdate


app.layout = html.Div([
    html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
    # dcc.Upload(
    #         html.Div([
    #         'Select a single or multi-view tiff to view ',
    #         html.A('Select Files')
    #     ]),
    #     id='upload-image',
    #     style={
    #         'width': '100%',
    #         'height': '60px',
    #         'lineHeight': '60px',
    #         'borderWidth': '1px',
    #         'borderStyle': 'dashed',
    #         'borderRadius': '5px',
    #         'textAlign': 'center',
    #         'margin': '10px'
    #     },
    #     # Allow multiple files to be uploaded
    #     multiple=False
    # ),
    du.Upload(
        id='upload-image',
        max_file_size=1800,  # 1800 Mb
        filetypes=['png', 'tif', 'tiff', 'csv'],
        upload_id="upload-image",
        # Unique session id
    ),
    dcc.Dropdown(id='image_layers'),
    html.Div(id='output_cell_image'),
    DashCanvas(id='image_annotation',
               tool='line',
               lineWidth=5,
               lineColor='red',
               width=1000)

])

