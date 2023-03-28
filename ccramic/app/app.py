import time

import anndata
import tifffile
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from PIL import Image, ImageSequence, ImageColor

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
import io
from flask_caching import Cache
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash_daq as daq
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import ctx, DiskcacheManager, CeleryManager
from tifffile import TiffFile
from matplotlib import pyplot as plt
from .utils import generate_tiff_stack, recolour_greyscale
import diskcache

app = DashProxy(transforms=[ServersideOutputTransform()], external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ccramic"

try:
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    })
except (ModuleNotFoundError, RuntimeError) as no_redis:
    # cache = Cache(app.server, config={
    #     'CACHE_TYPE': 'filesystem',
    #     'CACHE_DIR': 'cache-directory'
    # })
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

with tempfile.TemporaryDirectory() as tmpdirname:
    du.configure_upload(app, tmpdirname)


def convert_image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.callback(
    Output("metadata-distribution", "figure"),
    Input('anndata', 'data'),
    Input('metadata_options', 'value'))
def display_metadata_distribution(anndata_obj, metadata_selection):
    if anndata_obj is not None and metadata_selection is not None:
        data = anndata_obj['metadata'][metadata_selection]
        return px.histogram(data, range_x=[min(data), max(data)])
    else:
        raise PreventUpdate


@du.callback(ServersideOutput('uploaded_dict', 'data'),
             id='upload-image')
@cache.memoize()
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    upload_dict = {}
    if len(filenames) > 0:
        for upload in filenames:
            layer_index = 0
            file_designation = str(uuid.uuid4())
            if upload.endswith('.tiff') or upload.endswith('.tif'):
                # with TiffFile(upload) as tif:
                #     for page in tif.pages:
                #         # page = page.convert('RGB')
                #nimage = image.convert('RGB')
                upload_dict[os.path.basename(upload)] = tifffile.imread(upload)
            else:
                image = Image.open(upload)
                # image = image.convert('RGB')
                upload_dict[os.path.basename(upload)] = image

    if upload_dict:
        return upload_dict
    else:
        raise PreventUpdate


@du.callback(ServersideOutput('anndata', 'data'),
             id='upload-quantification')
@cache.memoize()
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    anndata_files = {}
    if filenames:
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


@du.callback(ServersideOutput('carousel_dict', 'data'),
             id='upload-image-carousel')
@cache.memoize()
def create_carousel_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    upload_dict = {}
    if len(filenames) > 0:
        for upload in filenames:
            layer_index = 0
            file_designation = str(uuid.uuid4())
            if upload.endswith('.tiff') or upload.endswith('.tif'):
                with TiffFile(upload) as tif:
                    for page in tif.pages:
                        upload_dict[file_designation + str(f"_{layer_index}")] = Image.fromarray(page.asarray())
                        layer_index += 1
            else:
                upload_dict[file_designation + str(f"_{layer_index}")] = Image.open(upload)

    if upload_dict:
        return upload_dict
    else:
        raise PreventUpdate


@app.callback(Output('dimension-reduction_options', 'options'),
              Input('anndata', 'data'))
def create_anndata_dimension_options(anndata_dict):
    if anndata_dict and "assays" in anndata_dict.keys():
        return [{'label': i, 'value': i} for i in anndata_dict["assays"].keys()]
    else:
        # image = Image.fromarray(tifffile.imread(upload))
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


@app.callback(Output('images_in_blend', 'options'),
              Input('image_layers', 'value'))
def create_dropdown_blend(chosen_for_blend):
    if chosen_for_blend:
        return [{'label': i, 'value': i} for i in chosen_for_blend]
    else:
        raise PreventUpdate


def read_back_base64_to_image(string):
    image_back = base64.b64decode(string)
    return Image.open(io.BytesIO(image_back))


@app.callback(Input("annotation-color-picker", 'value'),
              Input('images_in_blend', 'value'),
              Input('uploaded_dict', 'data'),
              Input('blending_colours', 'data'),
              Output('blending_colours', 'data'))
def set_blend_colour_for_layer(colour, layer, uploaded, current_blend_dict):
    if current_blend_dict is None and uploaded is not None:
        current_blend_dict = {}
        for uploaded in uploaded.keys():
            current_blend_dict[uploaded] = '#FFFFFF'
    if ctx.triggered_id == 'annotation-color-picker' and layer is not None:
        current_blend_dict[layer] = colour['hex']
        return current_blend_dict

    # elif ctx.triggered_id == 'image_layers' and blend_dict is not None:
    #     for cur in options:
    #         if cur not in blend_dict.keys():
    #             blend_dict[cur] = '#FFFFFF'
    #     return blend_dict
    else:
        return current_blend_dict


# @app.callback(Output("annotation-color-picker", 'value'),
#               Input('images_in_blend', 'value'),
#               Input('blending_colours', 'data'))
# def update_color_wheel(cur_selected, colour_dict):
#     if cur_selected is not None and cur_selected in colour_dict.keys():
#         return colour_dict[cur_selected].lower()
#     else:
#         raise PreventUpdate


@app.callback(Output('annotation_canvas', 'figure'),
              Input('image_layers', 'value'),
              Input('uploaded_dict', 'data'),
              Input('blending_colours', 'data'))
def render_image_on_canvas(image_str, image_dict, blend_colour_dict):
    if blend_colour_dict is None and image_str is not None:
        blend_colour_dict = {}
        for selected in image_str:
            blend_colour_dict[selected] = '#FFFFFF'
    if image_str is not None and len(image_str) <= 1:
            # (isinstance(image_str, list) and len(image_str) < 2):
        # image = Image.fromarray(image_dict[image_str[0]]).convert('RGB')
        image = recolour_greyscale(image_dict[image_str[0]], blend_colour_dict[image_str[0]])
        fig = px.imshow(image)
        return fig
    if image_str is not None and len(image_str) > 1:
        fig = px.imshow(generate_tiff_stack(image_dict, image_str, blend_colour_dict))
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


def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)



@du.callback(ServersideOutput('image-metadata', 'data'),
             id='upload-metadata')
@cache.memoize()
def create_imc_meta_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    imaging_metadata = {}
    if filenames:
        imc_frame = pd.read_csv(filenames[0])
        imaging_metadata["columns"] = [{'id': p, 'name': p} for p in imc_frame.columns]
        imaging_metadata["data"] = imc_frame.to_dict(orient='records')

        return imaging_metadata
    else:
        raise PreventUpdate


@app.callback(
    Output("imc-metadata-editable", "columns"),
    Output("imc-metadata-editable", "data"),
    Input('image-metadata', 'data'))
def populate_datatable_columns(column_dict):
    if column_dict is not None:
        return column_dict["columns"], column_dict["data"]
    else:
        raise PreventUpdate


@app.callback(
    Output("download-edited-table", "data"),
    Input("btn-download-metadata", "n_clicks"),
    Input("imc-metadata-editable", "data"),
    prevent_initial_call=True)
def download_edited_metadata(n_clicks, datatable_contents):
    if n_clicks is not None and n_clicks > 0 and datatable_contents is not None and \
            ctx.triggered_id == "btn-download-metadata":
        return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "metadata.csv")
    else:
        raise PreventUpdate


@app.callback(Output('blend-color-legend', 'children'),
              Input('blending_colours', 'data'),
              Input('images_in_blend', 'options'))
def create_legend(blend_colours, current_blend):
    children = []
    if blend_colours is not None and current_blend is not None:
        for key, value in blend_colours.items():
            if blend_colours[key] != '#FFFFFF':
                children.append(html.H6(f"{key}", style={"color": f"{value}"}))
        return html.Div(children=children)
    else:
        raise PreventUpdate


# @app.callback(Output('tiff-collage', 'items'),
#               Input('carousel_dict', 'data'),
#               background=True,
#               manager=background_callback_manager)
# def get_carousel_source_images(carousel_dict):
#     if carousel_dict is not None:
#         return [{"key": key, "src": value} for key, value in carousel_dict.items()]
#     else:
#         raise PreventUpdate
#
#
# @app.callback(Output('current-carousel-index', 'children'),
#               Input('tiff-collage', 'active_index'),
#               Input('carousel_dict', 'data'))
# def show_carousel_index(index, carousel_dict):
#     if index is not None and ctx.triggered_id == "tiff-collage":
#         return f"Current carousel: {list(carousel_dict.keys())[index]}"
#     else:
#         raise PreventUpdate


app.layout = html.Div([
    html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
    dcc.Tabs([
        dcc.Tab(label='Image Carousel', children=[
            du.Upload(
                id='upload-image-carousel',
                max_file_size=5000,
                filetypes=['tif', 'tiff'],
                upload_id="upload-image-carousel"),
            html.Div([dbc.Row([
                dbc.Col(html.Div([
                    dbc.Carousel(id='tiff-collage', items=[],
                                 style={'height': '35%', 'width': '35%'}, controls=True,
                                 indicators=True,
                                 interval=None),
                    html.Div(id='current-carousel-index', style={'whiteSpace': 'pre-line'})]),
                    width=6),
            ])]),

        ]),
        dcc.Tab(label='Image Annotation', children=[
            html.Div([dbc.Row([
                dbc.Col(html.Div([du.Upload(
                    id='upload-image',
                    max_file_size=5000,
                    max_files=200,
                    filetypes=['png', 'tif', 'tiff'],
                    upload_id="upload-image",
                ), dcc.Dropdown(id='image_layers', multi=True),
                    html.H3("Annotate your tif file"),
                    dash_table.DataTable(
                        id='imc-metadata-editable',
                        columns=[],
                        data=None,
                        editable=True),
                    dcc.Graph(config={
                        "modeBarButtonsToAdd": [
                            "drawline",
                            "drawopenpath",
                            "drawclosedpath",
                            "drawcircle",
                            "drawrect",
                            "eraseshape"]}, id='annotation_canvas', style={'width': '75vh', 'height': '75vh'}),
                    # html.Img(id='tiff_image', src=''),
                ]),
                    width=9),
                dbc.Col(html.Div([du.Upload(
                    id='upload-metadata',
                    max_file_size=1000,
                    max_files=1,
                    filetypes=['csv'],
                    upload_id="upload-image",
                ), html.Button("Download Edited metadata", id="btn-download-metadata"),
                    dcc.Download(id="download-edited-table"),
                    dcc.Dropdown(id='images_in_blend', multi=False),
                    daq.ColorPicker(id="annotation-color-picker",
                                    label="Color Picker", value=dict(hex="#119DFF")),
                html.Div(id='blend-color-legend', style={'whiteSpace': 'pre-line'})]), width=3),
            ])])
        ]),
        dcc.Tab(label='Quantification/Clustering', children=[
            du.Upload(
                id='upload-quantification',
                max_file_size=5000,
                filetypes=['h5ad', 'h5'],
                upload_id="upload-quantification"),
            html.Div([dbc.Row([
                dbc.Col(html.Div(["Dimension Reduction/Clustering",
                                  dcc.Dropdown(id='dimension-reduction_options'),
                                  dcc.Graph(id='umap-plot', style={'width': '150vh', 'height': '150vh'})]),
                        width=6),
                dbc.Col(html.Div(["Metadata Distribution",
                                  dcc.Dropdown(id='metadata_options'),
                                  dcc.Graph(id="metadata-distribution")]), width=6),
            ])]),

        ])
    ]),
    dcc.Loading(dcc.Store(id="uploaded_dict"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="blending_colours"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="anndata"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="image-metadata"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="carousel_dict"), fullscreen=True, type="dot")
])
