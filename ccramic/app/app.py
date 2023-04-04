
import anndata
import tifffile
import plotly.express as px
import pandas as pd
from PIL import Image
from dash import dash_table
import os
# from io import BytesIO
from dash.exceptions import PreventUpdate
import numpy as np
import flask
import tempfile
import dash_uploader as du
from flask_caching import Cache
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import ctx, DiskcacheManager
from tifffile import TiffFile
# from matplotlib import pyplot as plt
from .utils import generate_tiff_stack, recolour_greyscale
import diskcache
import h5py
import orjson

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
    upload_dict = {'single-channel': {}, 'multi-channel': {}}
    if len(filenames) > 0:
        for upload in filenames:
            if upload.endswith('.tiff') or upload.endswith('.tif'):
                if 'ome' not in upload:
                    upload_dict['single-channel'][os.path.basename(upload)] = tifffile.imread(upload)
                else:
                    with TiffFile(upload) as tif:
                        channel_index = 0
                        for page in tif.pages:
                            upload_dict['multi-channel'][str("channel_" + f"{channel_index}_") +
                                        os.path.basename(upload)] = (page.asarray() // 256).astype(np.uint8)
                            channel_index += 1.
            elif upload.endswith('.h5'):
                data_h5 = h5py.File(upload, "r")
                for image_type in list(data_h5.keys()):
                    for dataset in data_h5[image_type]:
                        upload_dict[image_type][dataset] = data_h5[image_type][dataset][()]
            else:
                image = Image.open(upload)
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


@app.callback(Output('dimension-reduction_options', 'options'),
              Input('anndata', 'data'))
def create_anndata_dimension_options(anndata_dict):
    if anndata_dict and "assays" in anndata_dict.keys():
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
              Input('uploaded_dict', 'data'),
              Input('tiff-image-type', 'value'))
def create_dropdown_options(image_dict, image_type):
    if image_dict and image_type:
        return [{'label': i, 'value': i} for i in image_dict[image_type].keys()]
    else:
        raise PreventUpdate


@app.callback(Output('images_in_blend', 'options'),
              Input('image_layers', 'value'))
def create_dropdown_blend(chosen_for_blend):
    if chosen_for_blend:
        return [{'label': i, 'value': i} for i in chosen_for_blend]
    else:
        raise PreventUpdate


@app.callback(Input("annotation-color-picker", 'value'),
              State('images_in_blend', 'value'),
              Input('uploaded_dict', 'data'),
              Input('blending_colours', 'data'),
              State('tiff-image-type', 'value'),
              Output('blending_colours', 'data'))
def set_blend_colour_for_layer(colour, layer, uploaded, current_blend_dict, image_type):
    if ctx.triggered_id == "uploaded_dict":
        if current_blend_dict is None and uploaded is not None:
            current_blend_dict = {'single-channel': {}, 'multi-channel': {}}
            for type in current_blend_dict.keys():
                for pot_layer in list(uploaded[type].keys()):
                    current_blend_dict[type][pot_layer] = '#FFFFFF'
            return current_blend_dict
        elif current_blend_dict is not None and uploaded is not None:
            for type in ['single-channel', 'multi-channel']:
                for pot_layer in list(uploaded[type].keys()):
                    current_blend_dict[type][pot_layer] = '#FFFFFF'
            return current_blend_dict
    if ctx.triggered_id == 'annotation-color-picker' and \
            layer is not None and current_blend_dict is not None and image_type is not None:
        current_blend_dict[image_type][layer] = colour['hex']
        return current_blend_dict
    else:
        return None


@app.callback(Output('annotation_canvas', 'figure'),
              Input('image_layers', 'value'),
              State('uploaded_dict', 'data'),
              Input('blending_colours', 'data'),
              State('tiff-image-type', 'value'),
              prevent_initial_call=True)
def render_image_on_canvas(image_str, image_dict, blend_colour_dict, image_type):
    if blend_colour_dict is None and image_str is not None and image_type is not None and \
            len(image_dict[image_type].keys()) > 0:
        blend_colour_dict = {'single-channel': {}, 'multi-channel': {}}
        for selected in image_str:
            if selected not in blend_colour_dict[image_type].keys():
                blend_colour_dict[image_type][selected] = '#ffffff'
    if image_str is not None and 1 >= len(image_str) > 0 and \
            len(image_dict[image_type].keys()) > 0:
        image = recolour_greyscale(image_dict[image_type][image_str[0]], blend_colour_dict[image_type][image_str[0]])
        fig = px.imshow(image)
        return fig
    if image_str is not None and len(image_str) > 1 and \
            len(image_dict[image_type].keys()) > 0:
        fig = generate_tiff_stack(image_dict[image_type], image_str, blend_colour_dict[image_type])
        return px.imshow(fig)
    else:
        raise PreventUpdate


@app.callback(Output('umap-plot', 'figure'),
              Input('anndata', 'data'),
              Input('metadata_options', 'value'),
              Input('dimension-reduction_options', 'value'))
def render_umap_plot(anndata_obj, metadata_selection, assay_selection):
    if anndata_obj and "assays" in anndata_obj.keys() and metadata_selection and assay_selection:
        data = anndata_obj["full_obj"]
        return px.scatter(data.obsm[assay_selection], x=0, y=1, color=data.obs[metadata_selection],
                          labels={'color': metadata_selection})
    else:
        raise PreventUpdate


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


@app.callback(Output('download-link', 'href'),
              [Input('uploaded_dict', 'data')])
def update_href(uploaded):
    if uploaded is not None:
        relative_filename = os.path.join(tmpdirname,
                                         'downloads',
                                         'data.h5')
        if not os.path.exists(os.path.join(tmpdirname, 'downloads')):
            os.makedirs(os.path.join(tmpdirname, 'downloads'))
        hf = h5py.File(relative_filename, 'w')
        groups = ['single-channel', 'multi-channel']
        for group in groups:
            hf.create_group(group)
            for key, value in uploaded[group].items():
                hf[group].create_dataset(key, data=value)
        hf.close()
        return str(relative_filename)


@app.callback(
    Output('annotation_canvas', 'style'),
    Input('annotation-canvas-size', 'value'))
def update_canvas_size(value):
    if value is not None:
        return {'width': f'{value}vh', 'height': f'{value}vh'}
    else:
        raise PreventUpdate


@app.callback(Output('image-gallery-row', 'children'),
              # Input('image-analysis', 'value'),
              State('uploaded_dict', 'data'),
              Input('tiff-image-type', 'value'))
@cache.memoize()
def create_image_grid(data, image_type):
    if data is not None and image_type is not None:
        row_children = []
        for chosen in list(data[image_type].keys()):
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody(html.P(chosen, className="card-text")),
                                                  dbc.CardImg(src=Image.fromarray(data[image_type][chosen]).convert('RGB'),
                                                              bottom=True)]), width=3))

        return row_children


@app.server.route(os.path.join(tmpdirname) + '/downloads/<path:path>')
def serve_static(path):
    return flask.send_from_directory(
        os.path.join(tmpdirname, 'downloads'), path)


@app.callback(Output('blend-color-legend', 'children'),
              Input('blending_colours', 'data'),
              Input('images_in_blend', 'options'),
              State('tiff-image-type', 'value'))
def create_legend(blend_colours, current_blend, image_type):
    current_blend = [elem['label'] for elem in current_blend] if current_blend is not None else None
    children = []
    if blend_colours is not None and current_blend is not None and image_type is not None:
        for key, value in blend_colours[image_type].items():
            if blend_colours[image_type][key] != '#FFFFFF' and key in current_blend:
                children.append(html.H6(f"{key}", style={"color": f"{value}"}))
        return html.Div(children=children)
    else:
        raise PreventUpdate


app.layout = html.Div([
    html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
    dcc.Tabs([
        dcc.Tab(label='Image Annotation', children=[
            html.Div([dcc.Tabs(id='image-analysis',
                               children=[dcc.Tab(label='Pixel Analysis', id='pixel-analysis',
                                                 children=[
                                                     html.Div([
                                                         dbc.Row([
                                                             dbc.Col(
                                                                 html.Div([
                                                                     du.Upload(
                                                                         id='upload-image',
                                                                         max_file_size=5000,
                                                                         max_files=200,
                                                                         filetypes=['png', 'tif',
                                                                                    'tiff', 'h5'],
                                                                         upload_id="upload-image"),
                                                                     html.Div([html.H5("Choose image type",
                                                                    style={'width': '35%', 'display': 'inline-block'}),
                                                                    html.H5("Choose image layers",
                                                                    style={'width': '65%', 'display': 'inline-block'}),
                                                                               dcc.Dropdown(
                                                                         id='tiff-image-type',
                                                                         multi=False,
                                                                        options=['single-channel', 'multi-channel'],
                                                                     style={'width': '30%', 'display': 'inline-block',
                                                                            'margin-right': '-30'}),
                                                                         dcc.Dropdown(
                                                                         id='image_layers',
                                                                         multi=True,
                                                                         style={'width': '70%', 'display': 'inline-block'})],
                                                                     style={'width': '125%', 'height': '125%',
                                                                            'display': 'inline-block', 'margin-left': '-30'}),
                                                                     dcc.Slider(50, 200, 10,
                                                                                value=120,
                                                                                id='annotation-canvas-size'),
                                                                     html.H3(
                                                                         "Annotate your tif file"),
                                                                     dcc.Graph(config={
                                                                         "modeBarButtonsToAdd": [
                                                                             "drawline",
                                                                             "drawopenpath",
                                                                             "drawclosedpath",
                                                                             "drawcircle",
                                                                             "drawrect",
                                                                             "eraseshape"]},
                                                                         id='annotation_canvas',
                                                                         style={'width': '120vh',
                                                                                'height': '120vh'}),
                                                                 ]), width=9),
                                                             dbc.Col(html.Div([
                                                                 dcc.Dropdown(id='images_in_blend',
                                                                              multi=False),
                                                                 daq.ColorPicker(
                                                                     id="annotation-color-picker",
                                                                     label="Color Picker",
                                                                     value=dict(hex="#119DFF")),
                                                                 html.Div(id='blend-color-legend',
                                                                          style={
                                                                              'whiteSpace': 'pre-line'}),
                                                                 html.A(id='download-link',
                                                                        children='Download File'),
                                                             ]), width=3),
                                                         ])])]),
                                         dcc.Tab(label="Image Gallery", id='gallery-tab',
                                                 children=[html.Div(id="image-gallery", children=[
                                                     dbc.Row(id="image-gallery-row")])]),
                                         dcc.Tab(label="Panel Metadata",
                                                 children=[html.Div([dbc.Row([
                                                     dbc.Col(html.Div([
                                                         dash_table.DataTable(
                                                             id='imc-metadata-editable',
                                                             columns=[],
                                                             data=None,
                                                             editable=True),
                                                     ]),
                                                         width=9),
                                                     dbc.Col(html.Div([du.Upload(
                                                         id='upload-metadata',
                                                         max_file_size=1000,
                                                         max_files=1,
                                                         filetypes=['csv'],
                                                         upload_id="upload-image",
                                                     ), html.Button("Download Edited metadata",
                                                                    id="btn-download-metadata"),
                                                         dcc.Download(
                                                             id="download-edited-table")]),
                                                         width=3),
                                                 ])])])])]),
        ], id='tab-annotation'),
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

        ], id='tab-quant')
    ]),
    dcc.Loading(dcc.Store(id="uploaded_dict"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="hdf5_obj"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="blending_colours"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="anndata"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="image-metadata"), fullscreen=True, type="dot")
])
