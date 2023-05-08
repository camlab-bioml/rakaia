import anndata
import tifffile
import plotly.express as px
import pandas as pd
from dash import dash_table
import os
# from io import BytesIO
from dash.exceptions import PreventUpdate
import flask
import tempfile
import dash_uploader as du
from flask_caching import Cache
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import ctx, DiskcacheManager
from tifffile import TiffFile, imwrite
# from matplotlib import pyplot as plt
from .utils import *
import diskcache
import h5py
from sqlite3 import DatabaseError
from readimc import MCDFile
import math
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from .parsers import *


def init_callbacks(dash_app, tmpdirname, cache):
    dash_app.config.suppress_callback_exceptions = True

    @dash_app.callback(
        Output("metadata-distribution", "figure"),
        Input('anndata', 'data'),
        Input('metadata_options', 'value'))
    def display_metadata_distribution(anndata_obj, metadata_selection):
        if anndata_obj is not None and metadata_selection is not None:
            data = anndata_obj['metadata'][metadata_selection]
            fig = px.histogram(data, range_x=[min(data), max(data)])
            # fig = go.Figure(fig, layout=dict(dragmode='rect'))
            return fig
        else:
            raise PreventUpdate

    @du.callback(Output('session_config', 'data'),
                 id='upload-image')
    # @cache.memoize()
    def get_session_uploads_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        session_config = {'uploads': []}
        for file in filenames:
            session_config['uploads'].append(file)
            return session_config
        else:
            raise PreventUpdate

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       State('read-filepath', 'value'),
                       Input('add-file-by-path', 'n_clicks'),
                       State('session_config', 'data'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def get_session_uploads_from_filepath(filepath, clicks, cur_session):
        if filepath is not None and clicks > 0:
            session_config = cur_session if cur_session is not None and \
                                            len(cur_session['uploads']) > 0 else {'uploads': []}
            if os.path.exists(filepath):
                session_config['uploads'].append(filepath)
                return session_config
            else:
                raise PreventUpdate

    @dash_app.callback(ServersideOutput('uploaded_dict', 'data'),
                       Input('session_config', 'data'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def create_upload_dict_from_filepath_string(session_dict):
        if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
            upload_dict, blend_dict = populate_upload_dict(session_dict['uploads'])
            return upload_dict
        else:
            raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options'),
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

    @dash_app.callback(Output('dimension-reduction_options', 'options'),
                       Input('anndata', 'data'))
    def create_anndata_dimension_options(anndata_dict):
        if anndata_dict and "assays" in anndata_dict.keys():
            return [{'label': i, 'value': i} for i in anndata_dict["assays"].keys()]
        else:
            raise PreventUpdate

    @dash_app.callback(Output('metadata_options', 'options'),
                       Input('anndata', 'data'))
    def create_anndata_dimension_options(anndata_dict):
        if anndata_dict and "metadata" in anndata_dict.keys():
            return [{'label': i, 'value': i} for i in anndata_dict["metadata"].columns]
        else:
            raise PreventUpdate

    @dash_app.callback(Output('image_layers', 'options'),
                       Input('uploaded_dict', 'data'),
                       Input('data-collection', 'value'))
    def create_dropdown_options(image_dict, data_selection):
        if image_dict and data_selection:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            return [{'label': i, 'value': i} for i in image_dict[exp][slide][acq].keys()]
        else:
            raise PreventUpdate

    @dash_app.callback(Output('images_in_blend', 'options'),
                       Input('image_layers', 'value'))
    def create_dropdown_blend(chosen_for_blend):
        if chosen_for_blend:
            return [{'label': i, 'value': i} for i in chosen_for_blend]
        else:
            raise PreventUpdate

    @dash_app.callback(Input("annotation-color-picker", 'value'),
                       State('images_in_blend', 'value'),
                       Input('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       Input('image_layers', 'value'),
                       State('canvas-layers', 'data'),
                       Output('blending_colours', 'data'),
                       ServersideOutput('canvas-layers', 'data'),
                       # Input('pixel-hist', 'figure'),
                       Input('pixel-hist', 'relayoutData'),
                       Input('bool-apply-filter', 'value'),
                       State('filter-type', 'value'),
                       State("kernel-val-filter", 'value'),
                       Input('session_config', 'data'),
                       prevent_initial_call=True)
    def set_blend_options_for_layer(colour, layer, uploaded, current_blend_dict, data_selection, add_to_layer,
                                    all_layers, hist_layout, filter_chosen, filter_name, filter_value,
                                    session_dict):
        # if data is uploaded, initialize the colour dict with white
        # do not update the layers if none have been selected

        # populate the blend dict from an h5 upload from a previous session
        if ctx.triggered_id == "session_config" and uploaded is not None:
            upload_dict, current_blend_dict = populate_upload_dict(session_dict['uploads'])
            if current_blend_dict is not None:
                return current_blend_dict, None
            else:
                current_blend_dict = create_new_blending_dict(uploaded)
                return current_blend_dict, None

        if ctx.triggered_id in ["uploaded_dict"] and ctx.triggered_id not in ['image-analysis']:
            if current_blend_dict is None and uploaded is not None:
                current_blend_dict = create_new_blending_dict(uploaded)
                return current_blend_dict, None
            if current_blend_dict is not None and uploaded is not None:
                for exp in uploaded.keys():
                    if "metadata" not in exp:
                        for slide in uploaded[exp].keys():
                            for acq in uploaded[exp][slide].keys():
                                for channel in uploaded[exp][slide][acq].keys():
                                    current_blend_dict[exp][slide][acq][channel] = {'color': None,
                                                                                    'x_lower_bound': None,
                                                                                    'x_upper_bound': None,
                                                                                    'y_ceiling': None,
                                                                                    'filter_type': None,
                                                                                    'filter_val': None}
                                    current_blend_dict[exp][slide][acq][channel]['color'] = '#FFFFFF'
                return current_blend_dict, None
            return current_blend_dict, None
        # if a new image is added to the layer, update the colour to white by default
        # update the layers with the colour
        if ctx.triggered_id in ["image_layers"] and add_to_layer is not None and \
                current_blend_dict is not None and ctx.triggered_id not in ['image-analysis']:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
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
                    current_blend_dict[exp][slide][acq][elem] = {'color': None,
                                                                 'x_lower_bound': None,
                                                                 'x_upper_bound': None,
                                                                 'y_ceiling': None,
                                                                 'filter_type': None,
                                                                 'filter_val': None}
                    current_blend_dict[exp][slide][acq][elem]['color'] = '#FFFFFF'
                if elem not in all_layers[exp][slide][acq].keys():
                    # create a nested dict with the image and all of the filters being used for it
                    all_layers[exp][slide][acq][elem] = np.array(recolour_greyscale(uploaded[exp][slide][acq][elem],
                                                                                    current_blend_dict[exp][slide][acq][
                                                                                        elem]['color'])).astype(
                        np.uint8)
            return current_blend_dict, all_layers
        # if the trigger is the colour wheel, update the specific layer with the colour chosen
        # update the layers with the colour
        if ctx.triggered_id in ['annotation-color-picker'] and \
                layer is not None and current_blend_dict is not None and data_selection is not None and \
                current_blend_dict is not None and ctx.triggered_id not in ['image-analysis']:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            current_blend_dict[exp][slide][acq][layer]['color'] = colour['hex']
            array = uploaded[exp][slide][acq][layer]

            # if upper and lower bounds have been set before for this layer, use them before recolouring

            if current_blend_dict[exp][slide][acq][layer]['x_lower_bound'] is not None and \
                    current_blend_dict[exp][slide][acq][layer]['x_upper_bound'] is not None:
                array = filter_by_upper_and_lower_bound(array,
                                                        float(current_blend_dict[exp][slide][acq][layer][
                                                                  'x_lower_bound']),
                                                        float(current_blend_dict[exp][slide][acq][layer][
                                                                  'x_upper_bound']))

            # if filters have been selected, apply them before recolouring

            all_layers[exp][slide][acq][layer] = np.array(recolour_greyscale(array,
                                                                             colour['hex'])).astype(np.uint8)
            return current_blend_dict, all_layers

        if ctx.triggered_id in ["bool-apply-filter"] and layer is not None and \
                current_blend_dict is not None and data_selection is not None and \
                current_blend_dict is not None and filter_value is not None and \
                ctx.triggered_id not in ['image-analysis']:

            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            array = uploaded[exp][slide][acq][layer]

            if current_blend_dict[exp][slide][acq][layer]['x_lower_bound'] is not None and \
                    current_blend_dict[exp][slide][acq][layer]['x_upper_bound'] is not None:
                array = filter_by_upper_and_lower_bound(array,
                                                        float(current_blend_dict[exp][slide][acq][layer][
                                                                  'x_lower_bound']),
                                                        float(current_blend_dict[exp][slide][acq][layer][
                                                                  'x_upper_bound']))

            if len(filter_chosen) > 0 and filter_name is not None:
                if filter_name == "median":
                    array = median_filter(array, int(filter_value))
                else:
                    array = gaussian_filter(array, int(filter_value))

                current_blend_dict[exp][slide][acq][layer]['filter_type'] = filter_name
                current_blend_dict[exp][slide][acq][layer]['filter_val'] = filter_value

            else:
                current_blend_dict[exp][slide][acq][layer]['filter_type'] = None
                current_blend_dict[exp][slide][acq][layer]['filter_val'] = None

            all_layers[exp][slide][acq][layer] = np.array(recolour_greyscale(array,
                                                                             current_blend_dict[exp][slide][acq][layer][
                                                                                 'color'])).astype(np.uint8)

            return current_blend_dict, all_layers

        # imp: the histogram will reset on a tab change, so ensure that a tab change won't reset the canvas
        if ctx.triggered_id in ["pixel-hist"] and \
                layer is not None and current_blend_dict is not None and data_selection is not None and \
                current_blend_dict is not None and ctx.triggered_id not in ['image-analysis'] and \
                hist_layout != {'autosize': True}:

            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            array = uploaded[exp][slide][acq][layer]

            # when shape is first added, these are the keys
            if 'shapes' in hist_layout.keys() and len(hist_layout['shapes']) > 0:
                lower_bound = hist_layout['shapes'][0]['x0']
                upper_bound = hist_layout['shapes'][0]['x1']
                y_ceiling = hist_layout['shapes'][0]['y0']
                array = filter_by_upper_and_lower_bound(array, lower_bound, upper_bound)

                current_blend_dict[exp][slide][acq][layer]['x_lower_bound'] = lower_bound
                current_blend_dict[exp][slide][acq][layer]['x_upper_bound'] = upper_bound
                current_blend_dict[exp][slide][acq][layer]['y_ceiling'] = y_ceiling

            # when an existing shape is moved, the keys change to this format
            elif 'shapes[0].x0' and 'shapes[0].x1' in hist_layout:
                lower_bound = hist_layout['shapes[0].x0']
                upper_bound = hist_layout['shapes[0].x1']
                y_ceiling = hist_layout['shapes[0].y0']
                array = filter_by_upper_and_lower_bound(array, lower_bound,
                                                        upper_bound)

                current_blend_dict[exp][slide][acq][layer]['x_lower_bound'] = lower_bound
                current_blend_dict[exp][slide][acq][layer]['x_upper_bound'] = upper_bound
                current_blend_dict[exp][slide][acq][layer]['y_ceiling'] = y_ceiling

            # if there is no shape, reset the bounds to None
            else:
                current_blend_dict[exp][slide][acq][layer]['x_lower_bound'] = None
                current_blend_dict[exp][slide][acq][layer]['x_upper_bound'] = None
                current_blend_dict[exp][slide][acq][layer]['y_ceiling'] = None

            # if filters have been selected, apply them as well

            if current_blend_dict[exp][slide][acq][layer]['filter_type'] is not None and \
                    current_blend_dict[exp][slide][acq][layer]['filter_val'] is not None:
                if current_blend_dict[exp][slide][acq][layer]['filter_type'] == "median":
                    array = median_filter(array, int(current_blend_dict[exp][slide][acq][layer]['filter_val']))
                else:
                    array = gaussian_filter(array, int(current_blend_dict[exp][slide][acq][layer]['filter_val']))

            # array = array * scale_factor
            all_layers[exp][slide][acq][layer] = np.array(recolour_greyscale(array,
                                                                             current_blend_dict[exp][slide][acq][layer][
                                                                                 'color'])).astype(np.uint8)

            return current_blend_dict, all_layers

        else:
            raise PreventUpdate

    @dash_app.callback(Output('image_layers', 'value'),
                       Input('data-collection', 'value'),
                       State('image_layers', 'value'),
                       prevent_initial_call=True)
    def reset_image_layers_selected(current_layers, new_selection):
        if new_selection is not None and current_layers is not None:
            if len(current_layers) > 0:
                return []
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure'),
                       Output('download-link-canvas-tiff', 'href'),
                       Input('canvas-layers', 'data'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input('alias-dict', 'data'),
                       State('annotation_canvas', 'figure'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('custom-scale-val', 'value'),
                       State('images_in_blend', 'options'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def render_image_on_canvas(canvas_layers, currently_selected, data_selection, blend_colour_dict, aliases,
                               cur_graph, cur_graph_layout, custom_scale_val, current_blend):

        if canvas_layers is not None and currently_selected is not None and blend_colour_dict is not None and \
                data_selection is not None and ctx.triggered_id not in ["annotation_canvas", "custom-scale-val",
                                                                        "image-analysis"]:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            legend_text = ''
            for image in currently_selected:
                if blend_colour_dict[exp][slide][acq][image]['color'] not in ['#ffffff', '#FFFFFF']:
                    label = aliases[image] if aliases is not None and image in aliases.keys() else image
                    legend_text = legend_text + f'<span style="color:' \
                                f'{blend_colour_dict[exp][slide][acq][image]["color"]}">{label}</span><br>'
            image = sum([np.asarray(canvas_layers[exp][slide][acq][elem]) for elem in currently_selected])
            try:
                fig = px.imshow(image)
                # set how far in from the lefthand corner the scale bar and colour legends should be
                # higher values mean closer to the centre
                x_axis_placement = 0.00001 * image.shape[1]
                # make sure the placement is min 0.05 and max 0.1
                x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.1 else 0.05
                # fig = canvas_layers[image_type][currently_selected[0]]
                if legend_text != '':
                    fig.add_annotation(text=legend_text, font={"size": 15}, xref='paper',
                                       yref='paper',
                                       x=(1 - x_axis_placement),
                                       # xanchor='right',
                                       y=0.05,
                                       # yanchor='bottom',
                                       showarrow=False)

                # set the x-axis scale placement based on the size of the image
                # for adding a scale bar
                fig.add_shape(type="line",
                              xref="paper", yref="paper",
                              x0=x_axis_placement, y0=0.05, x1=(x_axis_placement + 0.075),
                              y1=0.05,
                              line=dict(
                                  color="white",
                                  width=2,
                              ),
                              )
                # add annotation for the scaling
                # scale_val = int(custom_scale_val) if custom_scale_val is not None else
                scale_val = int(0.075 * image.shape[1])
                scale_annot = str(scale_val) + "um"
                scale_text = f'<span style="color: white">{scale_annot}</span><br>'
                fig.add_annotation(text=scale_text, font={"size": 10}, xref='paper',
                                   yref='paper',
                                   x=x_axis_placement + (0.075 / len(scale_annot)),
                                   # xanchor='right',
                                   y=0.06,
                                   # yanchor='bottom',
                                   showarrow=False)

                fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                  xaxis=go.XAxis(showticklabels=False),
                                  yaxis=go.YAxis(showticklabels=False),
                                  margin=dict(
                                      l=10,
                                      r=0,
                                      b=25,
                                      t=35,
                                      pad=0
                                  ))

                dest_file = os.path.join(tmpdirname, 'downloads', "canvas.tiff")
                if not os.path.exists(os.path.join(tmpdirname, 'downloads')):
                    os.makedirs(os.path.join(tmpdirname, 'downloads'))

                # fig_bytes = pio.to_image(fig, height=image.shape[1], width=image.shape[0])
                # buf = io.BytesIO(fig_bytes)
                # img = Image.open(buf)
                imwrite(dest_file, image, photometric='rgb')
                # plotly.offline.plot(fig, filename='tiff.html')
                # pio.write_image(fig, 'test_back.png', width=im.width, height=im.height)

                return fig, str(dest_file)
            except ValueError:
                raise PreventUpdate

        # update the scale bar with and without zooming
        elif ctx.triggered_id in ["annotation_canvas", "custom-scale-val"] and \
                cur_graph is not None and \
                'shapes' not in cur_graph_layout and ctx.triggered_id not in ["image-analysis"] and \
                cur_graph_layout != {"autosize": True}:
            # find the text annotation that has um in the text and the correct location
            for annotations in cur_graph['layout']['annotations']:
                if 'um' in annotations['text'] and annotations['y'] == 0.06:
                    if cur_graph_layout != {'autosize': True}:
                        x_range_high = 0
                        x_range_low = 0
                        # use different variables depending on how the ranges are written in the dict
                        # IMP: the variables will be written differently after a tab change
                        if 'xaxis' in cur_graph['layout']:
                            x_range_high = math.ceil(int(abs(cur_graph['layout']['xaxis']['range'][1])))
                            x_range_low = math.floor(int(abs(cur_graph['layout']['xaxis']['range'][0])))
                        elif 'xaxis.range[0]' and 'xaxis.range[1]' in cur_graph_layout:
                            x_range_high = math.ceil(int(abs(cur_graph_layout['xaxis.range[1]'])))
                            x_range_low = math.ceil(int(abs(cur_graph_layout['xaxis.range[1]'])))

                        assert x_range_high >= x_range_low
                        scale_val = int(custom_scale_val) if custom_scale_val is not None else \
                            int(math.ceil(int(0.075 * (x_range_high - x_range_low))) + 1)
                        scale_val = scale_val if scale_val > 0 else 1
                        scale_annot = str(scale_val) + "um"
                        scale_text = f'<span style="color: white">{str(scale_annot)}</span><br>'
                        # get the index of thre list element corresponding to this text annotation
                        index = cur_graph['layout']['annotations'].index(annotations)
                        cur_graph['layout']['annotations'][index]['text'] = scale_text

                        fig = go.Figure(cur_graph)
                        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                          xaxis=go.XAxis(showticklabels=False),
                                          yaxis=go.YAxis(showticklabels=False),
                                          margin=dict(
                                              l=25,
                                              r=0,
                                              b=25,
                                              t=35,
                                              pad=0
                                          ))
                    else:
                        fig = go.Figure(cur_graph)
                        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                          xaxis=go.XAxis(showticklabels=False, autorange=True),
                                          yaxis=go.YAxis(showticklabels=False, autorange=True),
                                          margin=dict(
                                              l=25,
                                              r=0,
                                              b=25,
                                              t=35,
                                              pad=0
                                          ))
            return fig, None
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'config'),
                       Input('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       prevent_initial_call=True)
    def set_graph_config(current_canvas, cur_canvas_layout):
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

        # only update the resolution if the zoom is not used
        if current_canvas is not None and 'range' in current_canvas['layout']['xaxis'] and \
                'range' in current_canvas['layout']['yaxis'] and \
                all([elem not in cur_canvas_layout for elem in zoom_keys]):
            config = {
                "modeBarButtonsToAdd": [
                    "drawline",
                    # "drawopenpath",
                    "drawclosedpath",
                    # "drawcircle",
                    "drawrect",
                    "eraseshape"],
                'toImageButtonOptions': {
                    'format': 'png',  # one of png, svg, jpeg, webp
                    'filename': 'canvas',
                    'height': int(current_canvas['layout']['yaxis']['range'][0]),
                    'width': int(current_canvas['layout']['xaxis']['range'][1]),
                    # 'scale': 2  # Multiply title/legend/axis/canvas sizes by this factor
                }
            }

            return config
        else:
            raise PreventUpdate

    @dash_app.callback(Output('umap-plot', 'figure'),
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

    @dash_app.callback(
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

    @dash_app.callback(
        Input("imc-metadata-editable", "data"),
        Output('alias-dict', 'data'))
    def create_channel_label_dict(metadata):
        if metadata is not None:
            alias_dict = {}
            for elem in metadata:
                alias_dict[elem['Channel Name']] = elem['Channel Label']
            return alias_dict

    @dash_app.callback(
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

    @dash_app.callback(Output('download-link', 'href'),
                       Input('uploaded_dict', 'data'),
                       Input('imc-metadata-editable', 'data'),
                       Input('blending_colours', 'data'))
    def update_download_href_h5(uploaded, metadata_sheet, blend_dict):
        if uploaded is not None:
            relative_filename = os.path.join(tmpdirname,
                                             'downloads',
                                             'data.h5')
            if not os.path.exists(os.path.join(tmpdirname, 'downloads')):
                os.makedirs(os.path.join(tmpdirname, 'downloads'))
            hf = h5py.File(relative_filename, 'w')
            for exp in list(uploaded.keys()):
                if 'metadata' in exp:
                    meta_to_write = pd.DataFrame(metadata_sheet) if metadata_sheet is not None else \
                        pd.DataFrame(uploaded['metadata'])
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
                                hf[exp][slide][acq].create_group(key)
                                if 'image' not in hf[exp][slide][acq][key]:
                                    hf[exp][slide][acq][key].create_dataset('image', data=value)
                                if blend_dict is not None:
                                    for blend_key, blend_val in blend_dict[exp][slide][acq][key].items():
                                        data_write = blend_val if blend_val is not None else "None"
                                        hf[exp][slide][acq][key].create_dataset(blend_key, data=data_write)

            hf.close()
            return str(relative_filename)

    @dash_app.callback(
        Output('annotation_canvas', 'style'),
        Input('annotation-canvas-size', 'value'),
        Input('annotation_canvas', 'figure'),
        State('annotation_canvas', 'relayoutData'))
    def update_canvas_size(value, current_canvas, cur_graph_layout):
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

        # only update the resolution if the zoom is not used
        if cur_graph_layout is not None and all([elem not in cur_graph_layout for elem in zoom_keys]):
            # if the current canvas is not None, update using the aspect ratio
            # otherwise, use aspect of 1
            if current_canvas is not None and 'range' in current_canvas['layout']['xaxis'] and \
                    'range' in current_canvas['layout']['yaxis']:
                # aspect ratio is width divided by height
                aspect_ratio = int(current_canvas['layout']['xaxis']['range'][1]) / \
                               int(current_canvas['layout']['yaxis']['range'][0])
            else:
                aspect_ratio = 1

            if value is not None:
                return {'width': f'{value * aspect_ratio}vh', 'height': f'{value}vh'}
            else:
                raise PreventUpdate

        elif value is not None and current_canvas is None:
            return {'width': f'{value}vh', 'height': f'{value}vh'}
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("selected-area-table", "data"),
        Input('annotation_canvas', 'figure'),
        Input('annotation_canvas', 'relayoutData'),
        State('uploaded_dict', 'data'),
        State('image_layers', 'value'),
        State('data-collection', 'value'),
        State('image-analysis', 'value'),
        State('alias-dict', 'data'),
        prevent_initial_call=True)
    def update_area_information(graph, graph_layout, upload, layers, data_selection, cur_tab, aliases_dict):
        # these range keys correspond to the zoom feature
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

        if graph is not None and graph_layout is not None and data_selection is not None and cur_tab == "tab-1":
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            # option 1: if shapes are drawn on the canvas
            if 'shapes' in graph_layout and len(graph_layout['shapes']) > 0:
                # these are for each sample
                mean_panel = []
                max_panel = []
                min_panel = []
                aliases = []
                for layer in layers:
                    try:
                        # for each layer we store the values for each shape
                        shapes_mean = []
                        shapes_max = []
                        shapes_min = []
                        for shape in graph_layout['shapes']:
                            # option 1: if the shape is drawn with a rectangle
                            if shape['type'] == 'rect':
                                x_range_low = math.ceil(int(shape['x0']))
                                x_range_high = math.ceil(int(shape['x1']))
                                y_range_low = math.ceil(int(shape['y0']))
                                y_range_high = math.ceil(int(shape['y1']))

                                assert x_range_high >= x_range_low
                                assert y_range_high >= y_range_low

                                mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[exp][slide][acq][layer],
                                                                                 x_range_low,
                                                                                 x_range_high,
                                                                                 y_range_low, y_range_high)
                                shapes_mean.append(round(float(mean_exp), 2))
                                shapes_max.append(round(float(max_xep), 2))
                                shapes_min.append(round(float(min_exp), 2))
                            # option 2: if a closed form shape is drawn
                            elif shape['type'] == 'path' and 'path' in shape:
                                mean_exp, max_xep, min_exp = get_area_statistics_from_closed_path(
                                    upload[exp][slide][acq][layer], shape['path'])
                                shapes_mean.append(round(float(mean_exp), 2))
                                shapes_max.append(round(float(max_xep), 2))
                                shapes_min.append(round(float(min_exp), 2))

                        mean_panel.append(round(sum(shapes_mean) / len(shapes_mean), 2))
                        max_panel.append(round(sum(shapes_max) / len(shapes_max), 2))
                        min_panel.append(round(sum(shapes_min) / len(shapes_min), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    except (AssertionError, ValueError, ZeroDivisionError):
                        pass

                layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}
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
                    aliases = []
                    for layer in layers:
                        mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[exp][slide][acq][layer], x_range_low,
                                                                         x_range_high,
                                                                         y_range_low, y_range_high)
                        mean_panel.append(round(float(mean_exp), 2))
                        max_panel.append(round(float(max_xep), 2))
                        min_panel.append(round(float(min_exp), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

                    return pd.DataFrame(layer_dict).to_dict(orient='records')

                except (AssertionError, ValueError, ZeroDivisionError):
                    return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                         'Min': []}).to_dict(orient='records')
            else:
                return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                     'Min': []}).to_dict(orient='records')
        else:
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')

    @dash_app.callback(Output('image-gallery-row', 'children'),
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
        else:
            raise PreventUpdate

    @dash_app.server.route(os.path.join(tmpdirname) + '/downloads/<path:path>')
    def serve_static(path):
        return flask.send_from_directory(
            os.path.join(tmpdirname, 'downloads'), path)

    @dash_app.callback(Output('blend-color-legend', 'children'),
                       Input('blending_colours', 'data'),
                       Input('images_in_blend', 'options'),
                       State('data-collection', 'value'),
                       Input('alias-dict', 'data'))
    def create_legend(blend_colours, current_blend, data_selection, aliases):
        current_blend = [elem['label'] for elem in current_blend] if current_blend is not None else None
        children = []
        if blend_colours is not None and current_blend is not None and data_selection is not None:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            for key, value in blend_colours[exp][slide][acq].items():
                if blend_colours[exp][slide][acq][key]['color'] != '#FFFFFF' and key in current_blend:
                    label = aliases[key] if aliases is not None and key in aliases.keys() else key
                    children.append(html.H6(f"{label}", style={"color": f"{value['color']}"}))

            return html.Div(children=children)
        else:
            raise PreventUpdate
    #
    # @dash_app.callback(Input('annotation_canvas', 'relayoutData'),
    #                    Output('get-polygon-coords', 'children'))
    # def print_polygon_coords(layout):
    #     if layout is not None:
    #         print(layout)

    @dash_app.callback(
        Output("download-collapse", "is_open"),
        [Input("open-download-collapse", "n_clicks")],
        [State("download-collapse", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @dash_app.callback(Output("pixel-hist", 'figure'),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'))
    def create_pixel_histogram(selected_channel, uploaded, data_selection, current_blend_dict):
        if None not in (selected_channel, uploaded, data_selection) and ctx.triggered_id not in ["image-analysis"]:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            # binwidth = 10
            # converted = Image.fromarray(uploaded[exp][slide][acq][selected_channel])
            # converted = np.array(converted, dtype=int)
            fig = pixel_hist_from_array(uploaded[exp][slide][acq][selected_channel])
            fig.update_layout(dragmode='drawrect')

            # if the current selection has already had a histogram bound on it, update the histogram with it
            if current_blend_dict[exp][slide][acq][selected_channel]['x_lower_bound'] is not None and \
                    current_blend_dict[exp][slide][acq][selected_channel]['x_upper_bound'] is not None:
                lower_bound = current_blend_dict[exp][slide][acq][selected_channel]['x_lower_bound']
                upper_bound = current_blend_dict[exp][slide][acq][selected_channel]['x_upper_bound']
                y_ceiling = current_blend_dict[exp][slide][acq][selected_channel]['y_ceiling']

                fig.add_shape(editable=True, type="rect", xref="x", yref="y", x0=lower_bound, y0=y_ceiling,
                              x1=upper_bound, y1=0,
                              line=dict(color='#444', width=4, dash='solid'),
                              fillcolor='rgba(0,0,0,0)', opacity=1)
            return fig
        else:
            raise PreventUpdate

    @dash_app.callback(Output('bool-apply-filter', 'value'),
                       Output('filter-type', 'value'),
                       Output('kernel-val-filter', 'value'),
                       Output("annotation-color-picker", 'value'),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'))
    def update_channel_filter_inputs(selected_channel, uploaded, data_selection, current_blend_dict):
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict):
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            if current_blend_dict[exp][slide][acq][selected_channel]['filter_type'] is not None and \
                    current_blend_dict[exp][slide][acq][selected_channel]['filter_val'] is not None and \
                    current_blend_dict[exp][slide][acq][selected_channel]['color'] is not None:
                return [' apply/refresh filter'], \
                       current_blend_dict[exp][slide][acq][selected_channel]['filter_type'], \
                       current_blend_dict[exp][slide][acq][selected_channel]['filter_val'], \
                       dict(hex=current_blend_dict[exp][slide][acq][selected_channel]['color'])
            else:
                return [], "median", 3, dict(hex="#1978B6")
        else:
            raise PreventUpdate


def init_dashboard(server):
    dash_app = DashProxy(__name__,
                         transforms=[ServersideOutputTransform()],
                         external_stylesheets=[dbc.themes.BOOTSTRAP],
                         server=server,
                         routes_pathname_prefix="/ccramic/")
    dash_app.title = "ccramic"
    server.config['APPLICATION_ROOT'] = "/ccramic"

    # VALID_USERNAME_PASSWORD_PAIRS = {
    #     'ccramic_user': 'ccramic'
    # }
    #
    # dash_auth.BasicAuth(
    #     dash_app,
    #     VALID_USERNAME_PASSWORD_PAIRS
    # )
    try:
        cache = Cache(dash_app.server, config={
            'CACHE_TYPE': 'redis',
            'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
        })
    except (ModuleNotFoundError, RuntimeError) as no_redis:
        try:
            cache = diskcache.Cache("./cache")
            background_callback_manager = DiskcacheManager(cache)
        except DatabaseError:
            cache = Cache(dash_app.server, config={
                'CACHE_TYPE': 'filesystem',
                'CACHE_DIR': 'cache-directory'
            })

    with tempfile.TemporaryDirectory() as tmpdirname:
        du.configure_upload(dash_app, tmpdirname)

    dash_app.layout = html.Div([
        html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
        dcc.Tabs([
            dcc.Tab(label='Image Annotation', children=[
                html.Div([dcc.Tabs(id='image-analysis',
                                   children=[dcc.Tab(label='Pixel Analysis',
                                                     id='pixel-analysis',
                                                     children=[html.Div([dbc.Row([dbc.Col(html.Div([
                                                         du.Upload(id='upload-image', max_file_size=10000,
                                                                   max_total_size=10000,
                                                                   max_files=200,
                                                                   filetypes=['png', 'tif', 'tiff', 'h5', 'mcd']),
                                                         dcc.Input(
                                                             id="read-filepath",
                                                             type="text",
                                                             placeholder="Add upload by file path (local runs only)",
                                                             value=None,
                                                         ),
                                                         dbc.Button("Add file by path",
                                                                    id="add-file-by-path",
                                                                    className="mb-3",
                                                                    color="primary", n_clicks=0,
                                                                    style={"margin-left": "20px",
                                                                           "margin-top": "10px"}),
                                                         html.Div([html.H5(
                                                             "Choose data collection",
                                                             style={'width': '35%',
                                                                    'display': 'inline-block'}),
                                                             html.H5("Choose channel image", style={'width': '65%',
                                                                                                    'display': 'inline-block'}),
                                                             dcc.Dropdown(id='data-collection', multi=False, options=[],
                                                                          style={'width': '30%',
                                                                                 'display': 'inline-block',
                                                                                 'margin-right': '-30'}),
                                                             dcc.Dropdown(id='image_layers', multi=True,
                                                                          style={'width': '70%', 'height': '100px',
                                                                                 'display': 'inline-block'})],
                                                             style={'width': '125%', 'height': '100%',
                                                                    'display': 'inline-block', 'margin-left': '-30'}),
                                                         dcc.Slider(50, 100, 5, value=75, id='annotation-canvas-size'),
                                                         html.Div([html.H3("Image/Channel Blending",
                                                                           style={
                                                                               "margin-right": "50px"}),
                                                                   html.Br(),
                                                                   ],
                                                                  style={"display": "flex", "width": "100%"}),
                                                         dcc.Graph(config={"modeBarButtonsToAdd": [
                                                             # "drawline",
                                                             # "drawopenpath",
                                                             "drawclosedpath",
                                                             # "drawcircle",
                                                             "drawrect",
                                                             "eraseshape"],
                                                             'toImageButtonOptions': {
                                                                 'format': 'png',
                                                                 'filename': 'canvas',
                                                                 'scale': 1}},
                                                             relayoutData={'autosize': True},
                                                             id='annotation_canvas', style={"margin-top": "-30px"})]),
                                                         width=8),
                                                         dbc.Col(html.Div([html.H5("Select channel to modify",
                                                                                   style={'width': '50%',
                                                                                          'display': 'inline-block'}),
                                                                           html.Abbr("\u2753",
                                                                                     title="Select a channel image to change colour or pixel intensity.",
                                                                                     style={'width': '5%',
                                                                                            'display': 'inline-block'}),
                                                                           dcc.Dropdown(id='images_in_blend',
                                                                                        multi=False),
                                                                           html.Br(),
                                                                           daq.ColorPicker(id="annotation-color-picker",
                                                                                           label="Color Picker",
                                                                                           value=dict(hex="#1978B6")),
                                                                           html.Br(),
                                                                           dcc.Graph(id="pixel-hist",
                                                                                     figure={'layout': {
                                                                                         'margin': dict(l=10, r=5, b=25,
                                                                                                        t=35, pad=2)}},
                                                                                     style={'width': '60vh',
                                                                                            'height': '30vh'},
                                                                                     config={"modeBarButtonsToAdd": [
                                                                                         "drawrect", "eraseshape"],
                                                                                         'modeBarButtonsToRemove': [
                                                                                             'zoom', 'pan']}),
                                                                           html.Br(),
                                                                           dcc.Checklist(
                                                                               options=[' apply/refresh filter'],
                                                                               value=[],
                                                                               id="bool-apply-filter"
                                                                           ),
                                                                           dcc.Dropdown(['median', 'gaussian'],
                                                                                        'median',
                                                                                        id='filter-type'),
                                                                           dcc.Input(
                                                                               id="kernel-val-filter",
                                                                               type="number",
                                                                               value=3,
                                                                           ),
                                                                           html.Br(),
                                                                           html.Br(),
                                                                           html.H6("Current canvas blend",
                                                                                   style={'width': '75%'}),
                                                                           html.Div(id='blend-color-legend',
                                                                                    style={'whiteSpace': 'pre-line'}),
                                                                           html.Br(),
                                                                           html.Br(),
                                                                           html.H6("Add custom scale value",
                                                                                   style={'width': '75%'}),
                                                                           dcc.Input(
                                                                               id="custom-scale-val",
                                                                               type="number",
                                                                               value=None,
                                                                           ),
                                                                           html.Br(),
                                                                           html.Br(),
                                                                           html.H6("Selection information",
                                                                                   style={'width': '75%'}),
                                                                           html.Div([dash_table.DataTable(
                                                                               id='selected-area-table',
                                                                               columns=[{'id': p, 'name': p} for p in
                                                                                        ['Channel', 'Mean', 'Max',
                                                                                         'Min']],
                                                                               data=None)], style={"width": "85%"}),
                                                                           html.Br(),
                                                                           html.Br(),
                                                                           dbc.Button("Show download links",
                                                                                      id="open-download-collapse",
                                                                                      className="mb-3",
                                                                                      color="primary", n_clicks=0),
                                                                           dbc.Tooltip(
                                                                               "Hover over this to get the download links.",
                                                                               target="open-download-collapse"),
                                                                           html.Div(dbc.Collapse(
                                                                               html.Div([html.A(id='download-link',
                                                                                                children='Download File'),
                                                                                         html.Br(),
                                                                                         html.A(
                                                                                             id='download-link-canvas-tiff',
                                                                                             children='Download Canvas as tiff')]),
                                                                               id="download-collapse", is_open=False),
                                                                               style={"minHeight": "100px"})]),
                                                                 width=4)])])]),

                                             dcc.Tab(label="Image Gallery", id='gallery-tab',
                                                     children=[html.Div(id="image-gallery", children=[
                                                         dbc.Row(id="image-gallery-row")])]),

                                             dcc.Tab(label="Panel Metadata", children=
                                             [html.Div([dbc.Row([
                                                 dbc.Col(html.Div([
                                                     dash_table.DataTable(id='imc-metadata-editable', columns=[],
                                                                          data=None,
                                                                          editable=True)]), width=9),
                                                 dbc.Col(html.Div([du.Upload(id='upload-metadata',
                                                                             max_file_size=1000, max_files=1,
                                                                             filetypes=['csv'],
                                                                             upload_id="upload-image"),
                                                                   html.Button("Download Edited metadata",
                                                                               id="btn-download-metadata"),
                                                                   dcc.Download(id="download-edited-table")]),
                                                         width=3)])])])])])], id='tab-annotation'),
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
        dcc.Loading(dcc.Store(id="session_config"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="hdf5_obj"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="blending_colours"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="anndata"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="image-metadata"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="canvas-layers"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="alias-dict"), fullscreen=True, type="dot")
    ])

    dash_app.enable_dev_tools(debug=True)

    init_callbacks(dash_app, tmpdirname, cache)

    return dash_app.server
