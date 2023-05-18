import anndata
import imageio.plugins.freeimage
import tifffile
import plotly.express as px
import pandas as pd
from dash import dash_table
import os
# from io import BytesIO
from dash.exceptions import PreventUpdate
import flask
import dash_uploader as du
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash_bootstrap_components as dbc
from dash import ctx, DiskcacheManager
from tifffile import TiffFile, imwrite
import math
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
            ann_data = anndata_obj['metadata'][metadata_selection]
            fig = px.histogram(ann_data, range_x=[min(ann_data), max(ann_data)])
            return fig
        else:
            raise PreventUpdate

    @du.callback(Output('session_config', 'data'),
                 id='upload-image')
    # @cache.memoize()
    def get_session_uploads_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        session_config = {'uploads': []}
        if filenames:
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
            # TODO: fix ability to read in multiple files at different times
            # session_config = cur_session if cur_session is not None and \
            #                                 len(cur_session['uploads']) > 0 else {'uploads': []}
            session_config = {'uploads': []}
            if os.path.exists(filepath):
                session_config['uploads'].append(filepath)
                return session_config
            else:
                raise PreventUpdate

    @dash_app.callback(ServersideOutput('uploaded_dict', 'data'),
                       Output('session_config', 'data', allow_duplicate=True),
                       Input('session_config', 'data'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def create_upload_dict_from_filepath_string(session_dict):
        if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
            upload_dict, blend_dict, unique_images = populate_upload_dict(session_dict['uploads'])
            session_dict['unique_images'] = unique_images
            return upload_dict, session_dict
        else:
            raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options'),
                       Output('data-collection', 'value'),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Input('uploaded_dict', 'data'),
                       prevent_initial_call=True)
    def populate_dataset_options(uploaded):
        # important: ensure that the selected value is reset to none when a new upload is made
        if uploaded is not None:
            datasets = []
            for exp in uploaded.keys():
                if "metadata" not in exp:
                    for slide in uploaded[exp].keys():
                        for acq in uploaded[exp][slide].keys():
                            datasets.append(f"{exp}_{slide}_{acq}")
            return datasets, None, None
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Input('uploaded_dict', 'data'),
                       State('annotation_canvas', 'figure'),
                       prevent_initial_call=True)
    def reset_canvas_on_new_upload(uploaded, cur_fig):
        if None not in (uploaded, cur_fig) and 'data' in cur_fig:
            return {}
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
                       Input('image_layers', 'value'),
                       prevent_initial_call=True)
    def create_dropdown_blend(chosen_for_blend):
        if chosen_for_blend is not None and len(chosen_for_blend) > 0:
            return [{'label': i, 'value': i} for i in chosen_for_blend]
        else:
            return []

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
                       Input('toggle-preset-use', 'value'),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'),
                       prevent_initial_call=True)
    def set_blend_options_for_layer(colour, layer, uploaded, current_blend_dict, data_selection, add_to_layer,
                                    all_layers, hist_layout, filter_chosen, filter_name, filter_value,
                                    session_dict, toggle_preset_use, preset_selection, preset_dict):

        # if data is uploaded, initialize the colour dict with white
        # do not update the layers if none have been selected

        # conditions where the callback should not occur
        # if the pixel hist is modified and the dragmode is either pan or zoom
        pixel_drag_changed = ctx.triggered_id in ["pixel-hist"] and hist_layout is not None and \
                             hist_layout in [{'dragmode': 'zoom'}, {'dragmode': 'pan'}]

        # if the callback is from the toggle preset and the current channel has the same parameters
        preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
        use_preset_condition = None not in (preset_selection, preset_dict) and toggle_preset_use

        if not pixel_drag_changed:
            # populate the blend dict from an h5 upload from a previous session
            if ctx.triggered_id == "session_config" and uploaded is not None:
                upload_dict, current_blend_dict, unique_images = populate_upload_dict(session_dict['uploads'])
                if current_blend_dict is None:
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
                    # if the selected channel doesn't have a config yet, create one either from scratch or a preset
                    if elem not in current_blend_dict[exp][slide][acq].keys():
                        current_blend_dict[exp][slide][acq][elem] = {'color': None,
                                                                     'x_lower_bound': None,
                                                                     'x_upper_bound': None,
                                                                     'y_ceiling': None,
                                                                     'filter_type': None,
                                                                     'filter_val': None}
                        current_blend_dict[exp][slide][acq][elem]['color'] = '#FFFFFF'
                        if use_preset_condition:
                            current_blend_dict[exp][slide][acq][elem] = apply_preset_to_blend_dict(
                                current_blend_dict[exp][slide][acq][elem], preset_dict[preset_selection])
                    # if the selected channel is in the current blend, check if a preset is used to override
                    elif elem in current_blend_dict[exp][slide][acq].keys() and use_preset_condition:
                        # do not override the colour of the curreht channel
                        current_blend_dict[exp][slide][acq][elem] = apply_preset_to_blend_dict(
                            current_blend_dict[exp][slide][acq][elem], preset_dict[preset_selection])
                    if elem not in all_layers[exp][slide][acq].keys():
                        # create a nested dict with the image and all of the filters being used for it
                        all_layers[exp][slide][acq][elem] = np.array(recolour_greyscale(uploaded[exp][slide][acq][elem],
                                                                                        current_blend_dict[exp][slide][
                                                                                            acq][
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

                if len(filter_chosen) > 0 and filter_name is not None:
                    if filter_name == "median":
                        array = median_filter(array, int(filter_value))
                    else:
                        array = gaussian_filter(array, int(filter_value))

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
                                                                                 current_blend_dict[exp][slide][acq][
                                                                                     layer][
                                                                                     'color'])).astype(np.uint8)

                return current_blend_dict, all_layers

            zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
            # imp: the histogram will reset on a tab change, so ensure that a tab change won't reset the canvas
            if ctx.triggered_id in ["pixel-hist"] and \
                    layer is not None and current_blend_dict is not None and data_selection is not None and \
                    current_blend_dict is not None and ctx.triggered_id not in ['image-analysis'] and \
                    hist_layout not in [{'autosize': True}, {'dragmode': 'zoom'}, {'dragmode': 'pan'}] and \
                    all([elem not in hist_layout for elem in zoom_keys]):

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
                                                                                 current_blend_dict[exp][slide][acq][
                                                                                     layer][
                                                                                     'color']))

                return current_blend_dict, all_layers
            # if a preset is selected, apply it to the current blend dict and array
            if ctx.triggered_id in ['toggle-preset-use', 'preset-options'] and None not in \
                    (preset_selection, preset_dict, data_selection, current_blend_dict, layer) and toggle_preset_use:
                split = data_selection.split("_")
                exp, slide, acq = split[0], split[1], split[2]
                array = uploaded[exp][slide][acq][layer]

                # preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
                for preset_val in preset_keys:
                    current_blend_dict[exp][slide][acq][layer][preset_val] = preset_dict[preset_selection][preset_val]

                array = apply_preset_to_array(array, preset_dict[preset_selection])
                all_layers[exp][slide][acq][layer] = np.array(recolour_greyscale(array,
                                                                                 current_blend_dict[exp][slide][acq][
                                                                                     layer][
                                                                                     'color']))
                return current_blend_dict, all_layers

            else:
                raise PreventUpdate
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
                       prevent_initial_call=True)
    # @cache.memoize()
    def render_image_on_canvas(canvas_layers, currently_selected, data_selection, blend_colour_dict, aliases,
                               cur_graph, cur_graph_layout, custom_scale_val):

        if canvas_layers is not None and currently_selected is not None and blend_colour_dict is not None and \
                data_selection is not None and ctx.triggered_id not in ["annotation_canvas", "custom-scale-val",
                                                                        "image-analysis"] and \
                len(currently_selected) > 0:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            legend_text = ''
            for image in currently_selected:
                if blend_colour_dict[exp][slide][acq][image]['color'] not in ['#ffffff', '#FFFFFF']:
                    label = aliases[image] if aliases is not None and image in aliases.keys() else image
                    legend_text = legend_text + f'<span style="color:' \
                                                f'{blend_colour_dict[exp][slide][acq][image]["color"]}"' \
                                                f'>{label}</span><br>'
            image = sum([np.asarray(canvas_layers[exp][slide][acq][elem]) for elem in currently_selected])
            try:
                fig = px.imshow(Image.fromarray(image))
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
                                       bgcolor="black",
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
                # this is the middle point of the scale bar
                # add shift based on the image shape
                shift = math.log10(image.shape[1]) - 3
                midpoint = (x_axis_placement + (0.075 / (2.5 * len(str(scale_val)) + shift)))
                fig.add_annotation(text=scale_text, font={"size": 10}, xref='paper',
                                   yref='paper',
                                   # set the placement of where the text goes relative to the scale bar
                                   x=midpoint,
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
                'shapes' not in cur_graph_layout and ctx.triggered_id not in ["image-analysis"]:
            try:
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

                        elif ctx.triggered_id == "custom-scale-val" and custom_scale_val is not None and \
                                cur_graph is not None:
                            scale_annot = str(custom_scale_val) + "um"
                            scale_text = f'<span style="color: white">{str(scale_annot)}</span><br>'
                            # get the index of the list element corresponding to this text annotation
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
            except ValueError:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'config'),
                       Input('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       prevent_initial_call=True)
    def set_graph_config(current_canvas, cur_canvas_layout):
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

        # only update the resolution if the zoom is not used
        try:
            if current_canvas is not None and 'range' in current_canvas['layout']['xaxis'] and \
                    'range' in current_canvas['layout']['yaxis'] and \
                    all([elem not in cur_canvas_layout for elem in zoom_keys]):
                config = {
                    'edits': {'shapePosition': False},
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
        except KeyError:
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
                                if blend_dict is not None and key in blend_dict[exp][slide][acq].keys():
                                    for blend_key, blend_val in blend_dict[exp][slide][acq][key].items():
                                        data_write = blend_val if blend_val is not None else "None"
                                        hf[exp][slide][acq][key].create_dataset(blend_key, data=data_write)

            hf.close()
            return str(relative_filename)
        else:
            raise PreventUpdate

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
            try:
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
                    return {'width': f'100vh', 'height': f'100vh'}
            except KeyError:
                return {'width': f'100vh', 'height': f'100vh'}
        elif value is not None and current_canvas is None:
            return {'width': f'{value}vh', 'height': f'{value}vh'}
        else:
            return {'width': f'100vh', 'height': f'100vh'}

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
        # these keys are used if a shape has been created, then modified
        modified_rect_keys = ['shapes[1].x0', 'shapes[1].x1', 'shapes[1].y0', 'shapes[1].y1']

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

                                mean_exp, max_xep, min_exp = get_area_statistics_from_rect(
                                    upload[exp][slide][acq][layer],
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
                        mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[exp][slide][acq][layer],
                                                                                   x_range_low,
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

            # option 3: if a shape has already been created and is modified
            elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                    all([elem in graph_layout for elem in modified_rect_keys]):
                try:
                    x_range_low = math.ceil(int(graph_layout['shapes[1].x0']))
                    x_range_high = math.ceil(int(graph_layout['shapes[1].x1']))
                    y_range_low = math.ceil(int(graph_layout['shapes[1].y0']))
                    y_range_high = math.ceil(int(graph_layout['shapes[1].y1']))
                    assert x_range_high >= x_range_low
                    assert y_range_high >= y_range_low

                    mean_panel = []
                    max_panel = []
                    min_panel = []
                    aliases = []
                    for layer in layers:
                        mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[exp][slide][acq][layer],
                                                                                   x_range_low,
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

            # option 4: if an svg path has already been created and it is modified
            elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                    all(['shapes' in elem and 'path' in elem for elem in graph_layout.keys()]):
                try:
                    mean_panel = []
                    max_panel = []
                    min_panel = []
                    aliases = []
                    for layer in layers:
                        for shape_path in graph_layout.values():
                            mean_exp, max_xep, min_exp = get_area_statistics_from_closed_path(
                                upload[exp][slide][acq][layer], shape_path)
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
                       Input('data-collection', 'value'),
                       State('canvas-layers', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('toggle-gallery-zoom', 'value'),
                       Input('toggle-preset-use', 'value'),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'),
                       Input('toggle-gallery-view', 'value'),
                       Input('unique-channel-list', 'value'))
    # @cache.memoize()
    def create_image_grid(gallery_data, data_selection, canvas_layers, canvas_layout, toggle_gallery_zoom,
                          toggle_preset_use, preset_selection, preset_dict, view_by_channel, channel_selected):
        if gallery_data is not None and gallery_data is not None:
            row_children = []
            zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

            # decide if channel view or ROI view is selected
            # channel view
            if view_by_channel and channel_selected is not None:
                views = get_all_images_by_channel_name(gallery_data, channel_selected)
            elif data_selection is not None:
                split = data_selection.split("_")
                exp, slide, acq = split[0], split[1], split[2]
                views = gallery_data[exp][slide][acq]
            else:
                views = None

            if views is not None:
                for key, value in views.items():
                    image_render = value

                    if toggle_preset_use and None not in (preset_selection, preset_dict):
                        image_render = apply_preset_to_array(value, preset_dict[preset_selection])

                    if all([elem in canvas_layout for elem in zoom_keys]) and toggle_gallery_zoom:
                        x_range_low = math.ceil(int(canvas_layout['xaxis.range[0]']))
                        x_range_high = math.ceil(int(canvas_layout['xaxis.range[1]']))
                        y_range_low = math.ceil(int(canvas_layout['yaxis.range[1]']))
                        y_range_high = math.ceil(int(canvas_layout['yaxis.range[0]']))
                        assert x_range_high >= x_range_low
                        assert y_range_high >= y_range_low
                        image_render = image_render[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                                                       range(int(x_range_low), int(x_range_high), 1))]

                    row_children.append(dbc.Col(dbc.Card([dbc.CardBody(html.P(key, className="card-text")),
                                                      dbc.CardImg(
                                                          src=resize_for_canvas(
                                                              Image.fromarray(
                                                                  image_render).
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
                       Input('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'))
    def create_pixel_histogram(selected_channel, uploaded, data_selection, current_blend_dict):
        if None not in (selected_channel, uploaded, data_selection) and ctx.triggered_id not in ["image-analysis"]:
            try:
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
                    fig.update_layout(showlegend=False)
                return fig
            except KeyError:
                return {}
        else:
            return {}

    @dash_app.callback(Output('bool-apply-filter', 'value'),
                       Output('filter-type', 'value'),
                       Output('kernel-val-filter', 'value'),
                       Output("annotation-color-picker", 'value'),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input('toggle-preset-use', 'value'),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'))
    def update_channel_filter_inputs(selected_channel, uploaded, data_selection, current_blend_dict,
                                     toggle_preset_use, preset_selection, preset_dict):
        """
        Update the input widgets wth the correct channel configs when the channel is changed, or a preset is used
        """
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict) and \
                ctx.triggered_id == "images_in_blend":
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            filter_type = current_blend_dict[exp][slide][acq][selected_channel]['filter_type']
            filter_val = current_blend_dict[exp][slide][acq][selected_channel]['filter_val']
            color = current_blend_dict[exp][slide][acq][selected_channel]['color']
            to_apply_filter = [' apply/refresh filter'] if None not in (filter_type, filter_val) else []
            filter_type_return = filter_type if filter_type is not None else "median"
            filter_val_return = filter_val if filter_val is not None else 3
            color_return = dict(hex=color) if color is not None and color not in ['#ffffff', '#FFFFFF'] \
                else dict(hex="#1978B6")
            return to_apply_filter, filter_type_return, filter_val_return, color_return
        elif ctx.triggered_id in ['toggle-preset-use', 'preset-options'] and None not in \
                (preset_selection, preset_dict, selected_channel, data_selection, current_blend_dict) and \
                toggle_preset_use:
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            filter_type = preset_dict[preset_selection]['filter_type']
            filter_val = preset_dict[preset_selection]['filter_val']
            color = current_blend_dict[exp][slide][acq][selected_channel]['color']
            to_apply_filter = [' apply/refresh filter'] if None not in (filter_type, filter_val) else []
            filter_type_return = filter_type if filter_type is not None else "median"
            filter_val_return = filter_val if filter_val is not None else 3
            color_return = dict(hex=color) if color is not None and color not in ['#ffffff', '#FFFFFF'] \
                else dict(hex="#1978B6")
            return to_apply_filter, filter_type_return, filter_val_return, color_return
        else:
            raise PreventUpdate

    @dash_app.callback(Input('preset-button', 'n_clicks'),
                       State('set-preset', 'value'),
                       State('preset-options', 'options'),
                       State('data-collection', 'value'),
                       State('images_in_blend', 'value'),
                       State('blending_colours', 'data'),
                       State('image_presets', 'data'),
                       Output('preset-options', 'options'),
                       Output('image_presets', 'data'))
    def generate_preset_options(selected_click, preset_name, current_preset_options, data_selection, layer,
                                current_blend_dict, current_presets):
        if selected_click is not None and selected_click > 0 and None not in (preset_name, data_selection, layer,
                                                                              current_blend_dict):
            split = data_selection.split("_")
            exp, slide, acq = split[0], split[1], split[2]
            if preset_name not in current_preset_options:
                current_preset_options.append(preset_name)

            current_presets = {} if current_presets is None else current_presets

            current_presets[preset_name] = current_blend_dict[exp][slide][acq][layer]

            return current_preset_options, current_presets
        else:
            raise PreventUpdate

    @dash_app.callback(Input('image_presets', 'data'),
                       Output('hover-preset-information', 'children'))
    def update_hover_preset_information(preset_dict):
        """
        Update the hover information on the list of presets so that the user can preview the parameters before selecting
        """
        if preset_dict is not None and len(preset_dict) > 0:
            text = ''
            for stud, val in preset_dict.items():
                text = text + f"{stud}: \r\n l_bound: {round(float(val['x_lower_bound']), 1)}, " \
                              f"y_bound: {round(float(val['x_upper_bound']), 1)}, filter type: {val['filter_type']}, " \
                              f"filter val: {val['filter_val']} \r\n"

            return html.Textarea(text, style={"width": "200px", "height": f"{100*len(preset_dict)}px"})
        else:
            raise PreventUpdate

    @dash_app.callback(Input('session_config', 'data'),
                       Output('unique-channel-list', 'options'))
    def populate_gallery_channel_list(session_config):
        """
        Populate a list of all unique channel names for the gallery view
        """
        if session_config is not None:
            try:
                return session_config['unique_images']
            except KeyError:
                return None
        else:
            raise PreventUpdate
