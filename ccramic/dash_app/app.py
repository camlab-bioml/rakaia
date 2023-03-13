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
import plotly.express as px
import io


app = Dash(__name__)
app.title = "ccramic"

CANVAS_CONFIG = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}

with tempfile.TemporaryDirectory() as tmpdirname:
    du.configure_upload(app, tmpdirname)


def convert_image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@du.callback(Output('uploaded_dict', 'data'),
              id='upload-image')
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
                upload_dict[file_designation + layer_designation] = convert_image_to_bytes(page)
                layer_index += 1
        if upload_dict:
            print(upload_dict)
            return upload_dict
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(Output('image_layers', 'options'),
              Input('uploaded_dict', 'data'))
def create_dropdown_options(image_dict):
    if image_dict is not None:
        return [{'label': i, 'value': i} for i in image_dict.keys()]
    else:
        raise PreventUpdate


def read_back_base64_to_image(string):
    image_back = base64.b64decode(string)
    return Image.open(io.BytesIO(image_back))


@app.callback(Output('annotation_canvas', 'figure'),
              Input('image_layers', 'value'),
              Input('uploaded_dict', 'data'))
def render_image_on_canvas(image_str, image_dict):
    if image_str is not None and image_str in image_dict.keys():
        return px.imshow(read_back_base64_to_image(image_dict[image_str]), aspect='auto')
    else:
        raise PreventUpdate


app.layout = html.Div([
    html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
    du.Upload(
        id='upload-image',
        max_file_size=1800,  # 1800 Mb
        filetypes=['png', 'tif', 'tiff', 'csv'],
        upload_id="upload-image",
        # Unique session id
    ),
    dcc.Dropdown(id='image_layers'),
    html.H3("Annotate your tif file"), dcc.Graph(config=CANVAS_CONFIG, id='annotation_canvas'),
    dcc.Store(id='uploaded_dict'),
    dcc.Store(id='uploaded_dict_paths'),

])

