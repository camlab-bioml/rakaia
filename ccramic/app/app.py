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
from dash import ctx, DiskcacheManager, Patch
from tifffile import TiffFile
# from matplotlib import pyplot as plt
from .utils import *
import diskcache
import h5py
import orjson
from sqlite3 import DatabaseError
from readimc import TXTFile, MCDFile
from io import BytesIO
import math
import plotly.graph_objects as go
from ccramic import __version__

app = DashProxy(transforms=[ServersideOutputTransform()], external_stylesheets=[dbc.themes.BOOTSTRAP],
                )
app.title = "ccramic"

try:
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    })
except (ModuleNotFoundError, RuntimeError) as no_redis:
    try:
        cache = diskcache.Cache("./cache")
        background_callback_manager = DiskcacheManager(cache)
    except DatabaseError:
        cache = Cache(app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'cache-directory'
        })

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
# @cache.memoize()
def create_layered_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    # upload_dict = {'single-channel': {}, 'multi-channel': {}, 'metadata': None}
    upload_dict = {}
    if len(filenames) > 0:
        for upload in filenames:
            # if reading back in data with h5
            if upload.endswith('.h5'):
                data_h5 = h5py.File(upload, "r")
                for exp in list(data_h5.keys()):
                    upload_dict[exp] = {}
                    if 'metadata' not in exp:
                        for slide in data_h5[exp].keys():
                            upload_dict[exp][slide] = {}
                            for acq in data_h5[exp][slide].keys():
                                upload_dict[exp][slide][acq] = {}
                                for channel in data_h5[exp][slide][acq]:
                                    upload_dict[exp][slide][acq][channel] = data_h5[exp][slide][acq][channel][()]
                    else:
                        meta_back = pd.DataFrame(data_h5['metadata'])
                        for col in meta_back.columns:
                            meta_back[col] = meta_back[col].str.decode("utf-8")
                        try:
                            meta_back.columns = [i.decode("utf-8") for i in data_h5['metadata_columns']]
                        except KeyError:
                            pass
                        upload_dict[exp] = meta_back
            else:
                experiment_index = 0
                upload_dict["experiment" + str(experiment_index)] = {}
                # slide_index = 0
                # acquisition_index = 0
                # if tiffs are uploaded, treat as one slide and one acquisition
                if upload.endswith('.tiff') or upload.endswith('.tif'):
                    upload_dict["experiment" + str(experiment_index)]["slide" + str(0)] = {}
                    upload_dict["experiment" + str(experiment_index)]["slide" + str(0)][
                        "acq" + str(0)] = {}
                    if 'ome' not in upload:
                        basename = str(upload).split(".tif")[0]
                        upload_dict["experiment" + str(experiment_index)]["slide" + str(0)]["acq" + str(0)][basename] = \
                            convert_to_below_255(tifffile.imread(upload))
                    else:
                        with TiffFile(upload) as tif:
                            basename = str(upload).split(".ome.tiff")[0]
                            channel_index = 0
                            for page in tif.pages:
                                upload_dict["experiment" + str(experiment_index)]["slide" +
                                                    str(0)]["acq" + str(0)][basename][basename +
                                                    str("channel_" + f"{channel_index}_") +
                                                         os.path.basename(upload)] = \
                                    convert_to_below_255(page.asarray())
                                channel_index += 1
                elif upload.endswith('.mcd'):
                    with MCDFile(upload) as mcd_file:
                        channel_names = None
                        channel_labels = None
                        slide_index = 0
                        for slide in mcd_file.slides:
                            upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)] = {}
                            acq_index = 0
                            for acq in slide.acquisitions:
                                upload_dict["experiment" + str(experiment_index)]["slide" + str(slide_index)][
                                    "acq" + str(acq_index)] = {}
                                if channel_labels is None:
                                    channel_labels = acq.channel_labels
                                    channel_names = acq.channel_names
                                    upload_dict['metadata'] = {'Cycle': range(1, len(channel_names) + 1, 1),
                                                           'Channel Name': channel_names,
                                                           'Channel Label': channel_labels}
                                    upload_dict['metadata_columns'] = ['Cycle', 'Channel Name', 'Channel Label']
                                else:
                                    assert all(label in acq.channel_labels for label in channel_labels)
                                    assert all(name in acq.channel_names for name in channel_names)
                                img = mcd_file.read_acquisition(acq)
                                channel_index = 0
                                for channel in img:
                                    upload_dict["experiment" + str(experiment_index)]["slide" +
                                                        str(slide_index)]["acq" +
                                                        str(acq_index)][channel_names[channel_index]] = channel
                                    channel_index += 1
                                acq_index += 1
                            slide_index += 1
                    experiment_index += 1
    if upload_dict:
        return upload_dict
    else:
        raise PreventUpdate


@app.callback(Output('data-collection', 'options'),
              Input('uploaded_dict', 'data'))
def populate_dataset_options(uploaded):
    if uploaded is not None:
        datasets = []
        for exp in uploaded.keys():
            if "metadata" not in exp:
                for slide in uploaded[exp].keys():
                    for acq in uploaded[exp][slide].keys():
                        datasets.append(f"{exp}_{slide}_{acq}")
        return datasets
    else:
        raise PreventUpdate
    

@du.callback(ServersideOutput('anndata', 'data'),
             id='upload-quantification')
# @cache.memoize()
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
              Input('data-collection', 'value'))
def create_dropdown_options(image_dict, data_selection):
    if image_dict and data_selection:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        return [{'label': i, 'value': i} for i in image_dict[exp][slide][acq].keys()]
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
              State('blending_colours', 'data'),
              State('data-collection', 'value'),
              Input('image_layers', 'value'),
              State('canvas-layers', 'data'),
              Output('blending_colours', 'data'),
              ServersideOutput('canvas-layers', 'data'),
              prevent_initial_call=True)
def set_blend_colour_for_layer(colour, layer, uploaded, current_blend_dict, data_selection, add_to_layer, all_layers):
    # if data is uploaded, initialize the colour dict with white
    # do not update the layers if none have been selected
    print(current_blend_dict)
    if ctx.triggered_id in ["uploaded_dict"]:
        if current_blend_dict is None and uploaded is not None:
            print("making new")
            current_blend_dict = {}
            for exp in uploaded.keys():
                if "metadata" not in exp:
                    current_blend_dict[exp] = {}
                    for slide in uploaded[exp].keys():
                        current_blend_dict[exp][slide] = {}
                        for acq in uploaded[exp][slide].keys():
                            current_blend_dict[exp][slide][acq] = {}
                            for channel in uploaded[exp][slide][acq].keys():
                                current_blend_dict[exp][slide][acq][channel] = '#FFFFFF'
            print(current_blend_dict)
            return current_blend_dict, None
        if current_blend_dict is not None and uploaded is not None:
            print("updating")
            for exp in uploaded.keys():
                if "metadata" not in exp:
                    for slide in uploaded[exp].keys():
                        for acq in uploaded[exp][slide].keys():
                            for channel in uploaded[exp][slide][acq].keys():
                                current_blend_dict[exp][slide][acq][channel] = '#FFFFFF'
            print(current_blend_dict)
            return current_blend_dict, None
        return current_blend_dict, None
    # if a new image is added to the layer, update the colour to white by default
    # update the layers with the colour
    if ctx.triggered_id == "image_layers" and add_to_layer is not None and current_blend_dict is not None:
        print(current_blend_dict)
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        print(exp, slide, acq)
        if all_layers is None:
            all_layers = {}
        if exp not in all_layers.keys():
            all_layers[exp] = {}
        if slide not in all_layers[exp].keys():
            all_layers[exp][slide] = {}
        if acq not in all_layers[exp][slide].keys():
            all_layers[exp][slide][acq] = {}
        for elem in add_to_layer:
            if elem not in current_blend_dict[exp][slide][acq].keys():
                current_blend_dict[exp][slide][acq][elem] = '#FFFFFF'
            if elem not in all_layers[exp][slide][acq].keys():
                all_layers[exp][slide][acq][elem] = np.array(recolour_greyscale(uploaded[exp][slide][acq][elem],
                                                                  '#FFFFFF')).astype(np.uint8)
        return current_blend_dict, all_layers
    # if the trigger is the colour wheel, update the specific layer with the colour chosen
    # update the layers with the colour
    if ctx.triggered_id == 'annotation-color-picker' and \
            layer is not None and current_blend_dict is not None and data_selection is not None and \
            current_blend_dict is not None:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        current_blend_dict[exp][slide][acq][layer] = colour['hex']
        all_layers[exp][slide][acq][layer] = np.array(recolour_greyscale(uploaded[exp][slide][acq][layer],
                                                          colour['hex'])).astype(np.uint8)
        return current_blend_dict, all_layers
    else:
        raise PreventUpdate


@app.callback(Output('image_layers', 'value'),
              Input('data-collection', 'value'),
              State('image_layers', 'value'),
              prevent_initial_call=True)
def reset_image_layers_selected(current_layers, new_selection):
    if new_selection is not None and current_layers is not None:
        if len(current_layers) > 0:
            return []
    else:
        raise PreventUpdate

@app.callback(Output('annotation_canvas', 'figure'),
              Input('canvas-layers', 'data'),
              State('image_layers', 'value'),
              State('data-collection', 'value'),
              State('blending_colours', 'data'),
              prevent_initial_call=True)
def render_image_on_canvas(canvas_layers, currently_selected, data_selection, blend_colour_dict):
    if canvas_layers is not None and currently_selected is not None and blend_colour_dict is not None and \
            data_selection is not None:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        legend_text = ''
        for image in currently_selected:
            if blend_colour_dict[exp][slide][acq][image] not in ['#ffffff', '#FFFFFF']:
                legend_text = legend_text + f'<span style="color:' \
                            f'{blend_colour_dict[exp][slide][acq][image]}">{image}</span><br>'
        image = sum([np.asarray(canvas_layers[exp][slide][acq][elem]) for elem in currently_selected])
        try:
            fig = px.imshow(image)
            # fig = canvas_layers[image_type][currently_selected[0]]
            if legend_text != '':
                fig.add_annotation(text=legend_text, font={"size": 15}, xref='paper',
                                   yref='paper',
                                   x=0.99,
                                   xanchor='right',
                                   y=0.1,
                                   yanchor='bottom',
                                   showarrow=False)

            # for adding a scale bar
            # fig.add_shape(
            #     type='line',
            #     x0=image.shape[0]*0.1, x1=image.shape[0]*0.1 + 50,
            #     y0=image.shape[0]*0.1, y1=image.shape[0]*0.1, line_color='white'
            # )

            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                              xaxis=go.XAxis(showticklabels=False),
                              yaxis=go.YAxis(showticklabels=False))
            return fig
        except ValueError:
            raise PreventUpdate
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
# @cache.memoize()
def create_imc_meta_dict(status: du.UploadStatus):
    filenames = [str(x) for x in status.uploaded_files]
    imaging_metadata = {}
    if filenames:
        imc_frame = pd.read_csv(filenames[0])
        imaging_metadata["columns"] = [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                                       p in imc_frame.columns]
        imaging_metadata["data"] = imc_frame.to_dict(orient='records')

        return imaging_metadata
    else:
        raise PreventUpdate


@app.callback(
    Output("imc-metadata-editable", "columns"),
    Output("imc-metadata-editable", "data"),
    Input('uploaded_dict', 'data'),
    Input('image-metadata', 'data'))
def populate_datatable_columns(uploaded, column_dict):
    if uploaded is not None and uploaded['metadata'] is not None:
        return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                p in uploaded['metadata'].keys()], \
               pd.DataFrame(uploaded['metadata']).to_dict(orient='records')
    elif column_dict is not None:
        return column_dict["columns"], column_dict["data"]
    else:
        raise PreventUpdate


@app.callback(
    Input("imc-metadata-editable", "data"),
    Output('alias-dict', 'data'))
def create_channel_label_dict(metadata):
    if metadata is not None:
        alias_dict = {}
        for elem in metadata:
            alias_dict[elem['Channel Name']] = elem['Channel Label']



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
        for exp in list(uploaded.keys()):
            if 'metadata' in exp:
                meta_to_write = pd.DataFrame(uploaded['metadata'])
                if 'columns' not in exp:
                    for col in meta_to_write:
                        meta_to_write[col] = meta_to_write[col].astype(str)
                    hf.create_dataset('metadata', data=meta_to_write.to_numpy())
                else:
                    hf.create_dataset('metadata_columns', data=meta_to_write.columns.values.astype('S'))
            else:
                hf.create_group(exp)
                for slide in uploaded[exp].keys():
                    hf[exp].create_group(slide)
                    for acq in uploaded[exp][slide].keys():
                        hf[exp][slide].create_group(acq)
                        for key, value in uploaded[exp][slide][acq].items():
                            hf[exp][slide][acq].create_dataset(key, data=value)
        hf.close()
        return str(relative_filename)


@app.callback(
    Output('annotation_canvas', 'style'),
    Input('annotation-canvas-size', 'value'),
    Input('annotation_canvas', 'figure'))
def update_canvas_size(value, current_canvas):
    if current_canvas is not None:
        # aspect ratio is width divided by height
        aspect_ratio = int(current_canvas['layout']['xaxis']['range'][1]) / \
                       int(current_canvas['layout']['yaxis']['range'][0])
    else:
        aspect_ratio = 1
    if value is not None:
        return {'width': f'{value*aspect_ratio}vh', 'height': f'{value}vh'}
    else:
        raise PreventUpdate


@app.callback(
    Output("selected-area-table", "data"),
    Input('annotation_canvas', 'figure'),
    Input('annotation_canvas', 'relayoutData'),
    State('uploaded_dict', 'data'),
    State('image_layers', 'value'),
    State('data-collection', 'value'))
def update_area_information(graph, graph_layout, upload, layers, data_selection):

    # these range keys correspond to the zoom feature
    zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

    if graph is not None and graph_layout is not None and data_selection is not None:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        # option 1: if shapes are drawn on the canvas
        if 'shapes' in graph_layout and len(graph_layout['shapes']) > 0:
            # these are for each sample
            mean_panel = []
            max_panel = []
            min_panel = []
            for layer in layers:
                # for each layer we store the values for each shape
                shapes_mean = []
                shapes_max = []
                shapes_min = []
                for shape in graph_layout['shapes']:
                    if shape['type'] == 'rect':
                        x_range_low = math.ceil(int(shape['x0']))
                        x_range_high = math.ceil(int(shape['x1']))
                        y_range_low = math.ceil(int(shape['y0']))
                        y_range_high = math.ceil(int(shape['y1']))

                        assert x_range_high >= x_range_low
                        assert y_range_high >= y_range_low

                        mean_exp, max_xep, min_exp = get_area_statistics(upload[exp][slide][acq][layer], x_range_low,
                                                                         x_range_high,
                                                                         y_range_low, y_range_high)
                        shapes_mean.append(round(float(mean_exp), 2))
                        shapes_max.append(round(float(max_xep), 2))
                        shapes_min.append(round(float(min_exp), 2))

                mean_panel.append(round(sum(shapes_mean) / len(shapes_mean), 2))
                max_panel.append(round(sum(shapes_max) / len(shapes_max), 2))
                min_panel.append(round(sum(shapes_min) / len(shapes_min), 2))

            layer_dict = {'Layer': layers, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}
            return pd.DataFrame(layer_dict).to_dict(orient='records')

        # option 2: if the zoom is used
        elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                all([elem in graph_layout for elem in zoom_keys]):
            try:
                x_range_low = math.ceil(int(graph_layout['xaxis.range[0]']))
                x_range_high = math.ceil(int(graph_layout['xaxis.range[1]']))
                y_range_low = math.ceil(int(graph_layout['yaxis.range[1]']))
                y_range_high = math.ceil(int(graph_layout['yaxis.range[0]']))
                assert x_range_high >= x_range_low
                assert y_range_high >= y_range_low

                mean_panel = []
                max_panel = []
                min_panel = []
                for layer in layers:
                    mean_exp, max_xep, min_exp = get_area_statistics(upload[exp][slide][acq][layer], x_range_low, x_range_high,
                                                                 y_range_low, y_range_high)
                    mean_panel.append(round(float(mean_exp), 2))
                    max_panel.append(round(float(max_xep), 2))
                    min_panel.append(round(float(min_exp), 2))

                layer_dict = {'Layer': layers, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

                return pd.DataFrame(layer_dict).to_dict(orient='records')

            except (AssertionError, ValueError):
                return pd.DataFrame({'Layer': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')
        else:
            return pd.DataFrame({'Layer': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')
    else:
        return pd.DataFrame({'Layer': [], 'Mean': [], 'Max': [],
                             'Min': []}).to_dict(orient='records')


@app.callback(Output('image-gallery-row', 'children'),
              # Input('image-analysis', 'value'),
              State('uploaded_dict', 'data'),
              Input('data-collection', 'value'))
# @cache.memoize()
def create_image_grid(data, data_selection):
    if data is not None and data is not None:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        row_children = []
        for chosen in list(data[exp][slide][acq].keys()):
            row_children.append(dbc.Col(dbc.Card([dbc.CardBody(html.P(chosen, className="card-text")),
                                                  dbc.CardImg(
                                                src=resize_for_canvas(
                                                    Image.fromarray(data[exp][slide][acq][chosen]).
                                                      convert('RGB')),
                                                      bottom=True)]), width=3))

        return row_children


@app.server.route(os.path.join(tmpdirname) + '/downloads/<path:path>')
def serve_static(path):
    return flask.send_from_directory(
        os.path.join(tmpdirname, 'downloads'), path)


@app.callback(Output('blend-color-legend', 'children'),
              Input('blending_colours', 'data'),
              Input('images_in_blend', 'options'),
              State('data-collection', 'value'))
def create_legend(blend_colours, current_blend, data_selection):
    current_blend = [elem['label'] for elem in current_blend] if current_blend is not None else None
    children = []
    if blend_colours is not None and current_blend is not None and data_selection is not None:
        split = data_selection.split("_")
        exp, slide, acq = split[0], split[1], split[2]
        for key, value in blend_colours[exp][slide][acq].items():
            if blend_colours[exp][slide][acq][key] != '#FFFFFF' and key in current_blend:
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
                                                                         max_file_size=10000,
                                                                         max_total_size=10000,
                                                                         max_files=200,
                                                                         filetypes=['png', 'tif',
                                                                                    'tiff', 'h5', 'mcd'],
                                                                         upload_id="upload-image"),
                                                                     html.Div([html.H5("Choose data collection",
                                                                                       style={'width': '35%',
                                                                                              'display': 'inline-block'}),
                                                                               html.H5("Choose channel image",
                                                                                       style={'width': '65%',
                                                                                    'display': 'inline-block'}),
                                                                               dcc.Dropdown(
                                                                                   id='data-collection',
                                                                                   multi=False,
                                                                                   options=[],
                                                                                   style={'width': '30%',
                                                                                          'display': 'inline-block',
                                                                                          'margin-right': '-30'}),
                                                                               dcc.Dropdown(
                                                                                   id='image_layers',
                                                                                   multi=True,
                                                                                   style={'width': '70%',
                                                                                          'height': '100px',
                                                                                          'display': 'inline-block'})],
                                                                              style={'width': '125%', 'height': '100%',
                                                                                     'display': 'inline-block',
                                                                                     'margin-left': '-30'}),
                                                                     dcc.Slider(50, 100, 10,
                                                                                value=75,
                                                                                id='annotation-canvas-size'),
                                                                     html.H3(
                                                                         "Annotate your tif file",
                                                                     style={"margin=bottom": "-30"}),
                                                                     dcc.Graph(config={
                                                                         "modeBarButtonsToAdd": [
                                                                             "drawline",
                                                                             "drawopenpath",
                                                                             "drawclosedpath",
                                                                             "drawcircle",
                                                                             "drawrect",
                                                                             "eraseshape"]},
                                                                         id='annotation_canvas',
                                                                         style={"margin-top": "-30"})
                                                                     # style={'width': '120vh',
                                                                     #        'height': '120vh'}),
                                                                 ]), width=8),
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
                                                                 html.Br(),
                                                                 html.Br(),
                                                                 # html.Div(id='selected-area-info',
                                                                 #       style={
                                                                 #           'whiteSpace': 'pre-line'}),
                                                                 html.Div([dash_table.DataTable(
                                                                     id='selected-area-table',
                                                                     columns=[{'id': p, 'name': p} for p in
                                                                              ['Layer', 'Mean', 'Max', 'Min']],
                                                                     data=None)], style={"width": "85%"}),
                                                             ]), width=4),
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
    dcc.Loading(dcc.Store(id="image-metadata"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="canvas-layers"), fullscreen=True, type="dot"),
    dcc.Loading(dcc.Store(id="alias-dict"), fullscreen=True, type="dot")
])
