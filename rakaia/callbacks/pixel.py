import os.path
import uuid
from pathlib import Path
import json
import dash.exceptions
import dash_uploader as du
import flask
from dash import ctx, ALL
from dash_extensions.enrich import Output, Input, State, html
from dash import dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from natsort import natsorted
import shortuuid
from rakaia.inputs.pixel import (
    wrap_canvas_in_loading_screen_for_large_images,
    invert_annotations_figure,
    set_range_slider_tick_markers,
    generate_canvas_legend_text,
    set_x_axis_placement_of_scalebar, update_canvas_filename,
    set_canvas_viewport, marker_correlation_children, reset_pixel_histogram)
from rakaia.parsers.pixel import (
    FileParser,
    populate_image_dict_from_lazy_load,
    create_new_blending_dict,
    populate_alias_dict_from_editable_metadata,
    check_blend_dictionary_for_blank_bounds_by_channel, check_empty_missing_layer_dict)
from rakaia.utils.decorator import (
    # time_taken_callback,
    DownloadDirGenerator)
from rakaia.utils.pixel import (
    delete_dataset_option_from_list_interactively,
    get_default_channel_upper_bound_by_percentile,
    apply_preset_to_array,
    recolour_greyscale,
    select_random_colour_for_channel,
    apply_preset_to_blend_dict,
    filter_by_upper_and_lower_bound,
    set_channel_list_order,
    pixel_hist_from_array,
    validate_incoming_metadata_table,
    make_metadata_column_editable,
    get_first_image_from_roi_dictionary,
    upper_bound_for_range_slider,
    no_filter_chosen,
    channel_filter_matches,
    ag_grid_cell_styling_conditions,
    MarkerCorrelation, high_low_values_from_zoom_layout, layers_exist, add_saved_blend)
from rakaia.utils.session import (
    validate_session_upload_config,
    channel_dropdown_selection,
    sleep_on_small_roi,
    set_data_selection_after_import)
from rakaia.components.canvas import CanvasImage, CanvasLayout, reset_graph_with_malformed_template
from rakaia.io.display import (
    RegionSummary,
    output_current_canvas_as_tiff,
    output_current_canvas_as_html,
    FullScreenCanvas,
    generate_preset_options_preview_text,
    annotation_preview_table, timestamp_download_child, generate_empty_region_table)
from rakaia.io.gallery import (
    generate_channel_tile_gallery_children,
    replace_channel_gallery_aliases)
from rakaia.parsers.object import ROIMaskMatch
from rakaia.utils.graph import strip_invalid_shapes_from_graph_layout
from rakaia.inputs.loaders import (
    previous_roi_trigger,
    next_roi_trigger,
    adjust_option_height_from_list_length, set_roi_tooltip_based_on_length, valid_key_trigger, mask_toggle_trigger)
from rakaia.callbacks.pixel_wrappers import parse_global_filter_values_from_json, parse_local_path_imports, \
    mask_options_from_json, bounds_text, generate_annotation_list, no_json_db_updates
from rakaia.io.session import (
    write_blend_config_to_json,
    write_session_data_to_h5py,
    subset_mask_for_data_export,
    SessionServerside, panel_match, all_roi_match, sort_channel_dropdown)
from rakaia.io.readers import DashUploaderFileReader
from rakaia.utils.db import (
    match_db_config_to_request_str,
    extract_alias_labels_from_db_document)
from rakaia.utils.alert import AlertMessage, file_import_message, DataImportError, LazyLoadError, \
    add_warning_to_error_config
from rakaia.utils.region import (
    RegionAnnotation,
    check_for_valid_annotation_hash)
from rakaia.parsers.roi import RegionThumbnail
from rakaia.utils.filter import (
    return_current_or_default_filter_apply,
    return_current_or_default_filter_param,
    return_current_channel_blend_params,
    return_current_or_default_channel_color,
    return_current_default_params_with_preset,
    apply_filter_to_channel, set_blend_parameters_for_channel)
from rakaia.callbacks.triggers import (
    no_canvas_mask,
    global_filter_disabled,
    channel_order_as_default,
    new_roi_same_dims,
    channel_already_added)


def init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with pixel level analysis/raw image preprocessing (image loading,
    blending, filtering, scaling, etc.)

    :param dash_app: the dash proxy server wrapped in the parent Flask app
    :param tmpdirname: the path for the tmpdir for tmp storage for the session
    :param authentic_id: uuid string identifying the particular app invocation
    :param app_config: Dictionary of session options passed through CLI
    :return: None
    """
    dash_app.config.suppress_callback_exceptions = True
    DEFAULT_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#00FAFF", "#FF00FF", "#FFFF00", "#FFFFFF"]
    ALERT = AlertMessage()
    ZOOM_KEYS = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
    OVERWRITE = app_config['serverside_overwrite']

    @du.callback(Output('uploads', 'data'),
                 id='upload-image')
    def get_filenames_from_drag_and_drop(status: du.UploadStatus):
        files = DashUploaderFileReader(status).return_filenames()
        return files if files else dash.no_update

    @du.callback(Output('param_blend_config', 'data', allow_duplicate=True),
                 id='upload-param-json')
    def get_param_json_from_drag_and_drop(status: du.UploadStatus):
        files = DashUploaderFileReader(status).return_filenames()
        return json.load(open(files[0])) if files else dash.no_update

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       Output('session_alert_config', 'data'),
                       State('read-filepath', 'value'),
                       Input('add-file-by-path', 'n_clicks'),
                       State('session_config', 'data'),
                       State('session_alert_config', 'data'),
                       prevent_initial_call=True)
    def get_session_uploads_from_local_path(path, clicks, cur_session, error_config):
        if path and clicks > 0:
            error_config = {"error": None} if error_config is None else error_config
            return parse_local_path_imports(path, validate_session_upload_config(cur_session), error_config)
        raise PreventUpdate

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       Input('local-dialog-file', 'n_clicks'),
                       State('session_config', 'data'),
                       prevent_initial_call=True)
    def read_from_local_dialog_box(nclicks, cur_session):
        if nclicks > 0:
            import wx
            app = wx.App(None)
            dialog = wx.FileDialog(None, 'Open', str(Path.home()), style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST,
                                   wildcard="*.tiff;*.tif;*.mcd;*.txt;*.h5|*.tiff;*.tif;*.mcd;*.txt;*.h5")
            if dialog.ShowModal() == wx.ID_OK:
                filenames = dialog.GetPaths()
                if filenames is not None and len(filenames) > 0 and isinstance(filenames, list):
                    session_config = validate_session_upload_config(cur_session)
                    for filename in filenames: session_config["uploads"].append(filename)
                    dialog.Destroy()
                    return session_config
                dialog.Destroy()
                raise PreventUpdate
            dialog.Destroy()
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       Input('uploads', 'data'),
                       State('session_config', 'data'),
                       prevent_initial_call=True)
    def populate_session_uploads_from_drag_and_drop(upload_list, cur_session):
        """
        populate the session uploads list from the list of uploads from a given execution.
        Requires the intermediate list as the callback is restricted to one output
        """
        session_config = validate_session_upload_config(cur_session)
        if upload_list is not None and len(upload_list) > 0:
            for new_upload in upload_list:
                if new_upload not in session_config["uploads"]: session_config["uploads"].append(new_upload)
            return session_config
        raise PreventUpdate

    @dash_app.callback(Output('uploaded_dict_template', 'data'),
                       Output('session_config', 'data', allow_duplicate=True),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('dataset-preview-table', 'columns'),
                       Output('dataset-preview-table', 'data'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('session_config', 'data'),
                       State('blending_colours', 'data'),
                       State('session_alert_config', 'data'),
                       State('natsort-uploads', 'value'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    def create_upload_dict_from_filepath_string(session_dict, current_blend, error_config, natsort, delimiter):
        """
        Create session variables from the list of imported file paths
        Note that a message will be supplied if more than one type of file is passed
        The image dictionary template is used to populate the actual images using lazy load
        """
        if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
            files = natsorted(session_dict['uploads']) if natsort else session_dict['uploads']
            message, unique_suffixes = file_import_message(files)
            suffix_add = ALERT.warnings["multiple_filetypes"] if len(unique_suffixes) > 1 else ""
            error_config = add_warning_to_error_config(error_config, suffix_add + message)
            try:
                fileparser = FileParser(files, array_store_type=app_config['array_store_type'], delimiter=delimiter)
                session_dict['unique_images'] = fileparser.unique_image_names
                columns = [{'id': p, 'name': p, 'editable': False} for p in fileparser.dataset_information_frame.keys()]
                data = pd.DataFrame(fileparser.get_parsed_information()).to_dict(orient='records')
                blend_return = fileparser.blend_config if (current_blend is None or len(current_blend) == 0) else dash.no_update
            except Exception as e:
                error_config = add_warning_to_error_config(error_config, str(e))
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, error_config
            return SessionServerside(fileparser.image_dict, key="upload_dict",
                                     use_unique_key=OVERWRITE), session_dict, blend_return, columns, data, error_config
        raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options'),
                       Output('data-collection', 'value'),
                       Output('image_layers', 'value'),
                       Output('data-collection', 'optionHeight'),
                       # Output('data-collection', 'className'),
                       Input('uploaded_dict_template', 'data'),
                       State('data-collection', 'value'),
                       State('image_layers', 'value'),
                       prevent_initial_call=True)
    def populate_dataset_options(uploaded, cur_data_selection, cur_layers_selected):
        if uploaded is not None:
            datasets, selection_return, channels_return = [], None, None
            for roi in uploaded.keys():
                if "metadata" not in roi: datasets.append(roi)
            if cur_data_selection is not None:
                selection_return = set_data_selection_after_import(datasets, cur_data_selection)
                if cur_layers_selected is not None and len(cur_layers_selected) > 0: channels_return = cur_layers_selected
            height_update = adjust_option_height_from_list_length(datasets)
            return datasets, selection_return, channels_return, height_update
            # can use an animation to draw attention to the data selection input
            # "animate__animated animate__jello animate__slower"
        raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options', allow_duplicate=True),
                       Output('data-collection', 'value', allow_duplicate=True),
                       Output('image_layers', 'options', allow_duplicate=True),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('dataset-preview-table', 'data', allow_duplicate=True),
                       Input('remove-collection', 'n_clicks'),
                       State('data-collection', 'value'),
                       State('data-collection', 'options'),
                       State('dataset-preview-table', 'data'),
                       prevent_initial_call=True)
    def remove_dataset_from_collection(remove_clicks, cur_data_selection, cur_options, data_preview):
        """
        Use the trash icon to remove a dataset collection from the possible selections
        Causes a reset of the canvas, channel selection, channel modification menus. and the data preview table
        """
        return delete_dataset_option_from_list_interactively(remove_clicks, cur_data_selection, cur_options, data_preview)

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Input('uploaded_dict_template', 'data'),
                       State('annotation_canvas', 'figure'),
                       prevent_initial_call=True)
    def reset_canvas_on_new_upload(uploaded, cur_fig):
        if None not in (uploaded, cur_fig) and 'data' in cur_fig and cur_fig['data']: return go.Figure().to_dict()
        raise PreventUpdate

    @dash_app.callback(Output('image_layers', 'options'),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('uploaded_dict', 'data', allow_duplicate=True),
                       Output('canvas-div-holder', 'children'),
                       Output('data-collection-tooltip', 'children'),
                       Output('cur_roi_dimensions', 'data'),
                       State('uploaded_dict_template', 'data'),
                       Input('data-collection', 'value'),
                       Input('alias-dict', 'data'),
                       State('image_layers', 'value'),
                       State('session_config', 'data'),
                       Input('sort-channels-alpha', 'value'),
                       State('enable-canvas-scroll-zoom', 'value'),
                       State('cur_roi_dimensions', 'data'),
                       State('dataset-delimiter', 'value'),
                       Input('data-selection-refresh', 'n_clicks'),
                       prevent_initial_call=True)
    def create_dropdown_options(image_dict, data_selection, names, currently_selected_channels, session_config,
                                sort_channels, enable_zoom, cur_dimensions, delimiter, refresh):
        """
        Update the image layers and dropdown options when a new ROI is selected.
        Additionally, check the dimension of the incoming ROI, and wrap the annotation canvas in a load screen
        if the dimensions are above a specific pixel height and width for either axis
        """
        # set the default canvas to return without a load screen
        if image_dict and data_selection and names:
            channels_return = sort_channel_dropdown(names, sort_channels)
            if ctx.triggered_id not in ["sort-channels-alpha", "alias-dict"]:
                try:
                    image_dict = populate_image_dict_from_lazy_load(image_dict.copy(), dataset_selection=data_selection,
                    session_config=session_config, array_store_type=app_config['array_store_type'], delimiter=delimiter)
                    if all([elem is None for elem in image_dict[data_selection].values()]):
                        raise LazyLoadError(AlertMessage().warnings["lazy-load-error"])
                    # check if the first image has dimensions greater than 3000. if yes, wrap the canvas in a loader
                    if data_selection in image_dict.keys() and all([image_dict[data_selection][elem] is not None for
                                                                    elem in image_dict[data_selection].keys()]):
                        # get the first image in the ROI and check the dimensions
                        first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
                        dim_return = (first_image.shape[0], first_image.shape[1])
                        # add a pause if the roi is really small to allow a full canvas update
                        sleep_on_small_roi(dim_return)
                        # if the new dimensions match, do not update the canvas child to preserve the ui revision state
                        if new_roi_same_dims(ctx.triggered_id, cur_dimensions, first_image):
                            canvas_return = dash.no_update
                        else:
                            canvas_return = [wrap_canvas_in_loading_screen_for_large_images(first_image, enable_zoom=
                            enable_zoom, wrap=app_config['use_loading'], filename=data_selection, delimiter=delimiter)]
                    else:
                        canvas_return = [wrap_canvas_in_loading_screen_for_large_images(None, enable_zoom=enable_zoom,
                                        wrap=app_config['use_loading'], filename=data_selection, delimiter=delimiter)]

                    # if all of the currently selected channels are in the new ROI, keep them. otherwise, reset
                    if currently_selected_channels is not None and len(currently_selected_channels) > 0 and \
                            all([elem in image_dict[data_selection].keys() for elem in currently_selected_channels]):
                        channels_selected = list(currently_selected_channels)
                    else:
                        channels_selected = []
                    return channel_dropdown_selection(channels_return, names), channels_selected, SessionServerside(
                        image_dict, key="upload_dict", use_unique_key=OVERWRITE), \
                        canvas_return, set_roi_tooltip_based_on_length(data_selection, delimiter), dim_return
                except Exception:
                    canvas_return = [wrap_canvas_in_loading_screen_for_large_images(None, enable_zoom=enable_zoom,
                                    wrap=app_config['use_loading'], filename=data_selection, delimiter=delimiter)]
                    return [], [], SessionServerside(image_dict, key="upload_dict", use_unique_key=OVERWRITE), \
                        canvas_return, set_roi_tooltip_based_on_length(data_selection, delimiter), dim_return
            elif ctx.triggered_id in ["sort-channels-alpha", "alias-dict"] and names is not None:
                return channel_dropdown_selection(channels_return, names), dash.no_update, dash.no_update, \
                    dash.no_update, dash.no_update, dash.no_update
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'config'),
                       Input('data-collection', 'value'),
                       State('annotation_canvas', 'config'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    def update_roi_download_name(roi_change, current_canvas_config, delim):
        """
        Update the name of the file output in the graph to match the ROI name
        """
        if roi_change and current_canvas_config: return update_canvas_filename(current_canvas_config, roi_change, delim)
        return current_canvas_config

    @dash_app.callback(Output('channel-quantification-list', 'options'),
                       Output('channel-quantification-list', 'value'),
                       Output('baseline-channel-cor', 'options'),
                       Output('target-channel-cor', 'options'),
                       Input('image_layers', 'options'),
                       Input('alias-dict', 'data'),
                       State('channel-quantification-list', 'value'),
                       Input('quant-toggle-list', 'value'),
                       prevent_initial_call=True)
    def create_channel_options_for_quantification_correlation(channel_options,
                                                              aliases, cur_selection, toggle_channels_quant):
        """
        Create the dropdown options for the channels for quantification and marker correlation
        If channels are already selected, keep them and just update the labels
        """
        channel_list_options = [{'label': value, 'value': key} for key, value in aliases.items()] if aliases else []
        channel_list_selected = list(aliases.keys()) if ctx.triggered_id == 'alias-dict' else cur_selection
        if ctx.triggered_id == 'quant-toggle-list':
            channel_list_selected = [elem['value'] for elem in channel_list_options] if toggle_channels_quant else []
        return channel_list_options, channel_list_selected, channel_list_options, channel_list_options

    @dash_app.callback(Output('images_in_blend', 'options'),
                       Output('images_in_blend', 'value'),
                       Input('image_layers', 'value'),
                       Input('alias-dict', 'data'),
                       State('images_in_blend', 'value'),
                       prevent_initial_call=True)
    def create_dropdown_blend(chosen_for_blend, names, cur_channel_mod):
        """
        Create the dropdown menu for the channel modification menu on layer changes
        Auto-fill the value with the latest channel if it doesn't match the current value in the modification
        """
        if chosen_for_blend is not None and len(chosen_for_blend) > 0:
            try:
                if not all([elem in names.keys() for elem in chosen_for_blend]): raise AssertionError
                channel_auto_fill = dash.no_update
                if chosen_for_blend[-1] != cur_channel_mod: channel_auto_fill = chosen_for_blend[-1]
                return [{'label': names[i], 'value': i} for i in chosen_for_blend], channel_auto_fill
            except (AssertionError, IndexError):
                raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Input('session_config', 'data'),
                       State('uploaded_dict_template', 'data'),
                       State('blending_colours', 'data'),
                       Output('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       prevent_initial_call=True)
    def create_new_blend_dict_on_upload(session_config, uploaded, current_blend_dict, current_selection):
        """
        Create a new blending dictionary on a new dataset upload.
        """
        if session_config is not None:
            if current_blend_dict is None and uploaded is not None and current_selection is None:
                return create_new_blending_dict(uploaded)
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(State('uploaded_dict', 'data'),
                       Input('param_blend_config', 'data'),
                       Input('db-config-options', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       State('session_alert_config', 'data'),
                       State('db-saved-configs', 'data'),
                       State("imc-panel-editable", "data"),
                       State('dataset-delimiter', 'value'),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('bool-apply-global-filter', 'value', allow_duplicate=True),
                       Output('global-filter-type', 'value', allow_duplicate=True),
                       Output("global-kernel-val-filter", 'value', allow_duplicate=True),
                       Output("global-sigma-val-filter", 'value', allow_duplicate=True),
                       Output("imc-panel-editable", "data", allow_duplicate=True),
                       Output('db-config-options', 'value', allow_duplicate=True),
                       Output('db-config-name', 'value', allow_duplicate=True),
                       Output('cluster-colour-assignments-dict', 'data', allow_duplicate=True),
                       Output('gating-dict', 'data', allow_duplicate=True),
                       Output('images_in_blend', 'value', allow_duplicate=True),
                       Output('apply-mask', 'value', allow_duplicate=True),
                       Output('mask-blending-slider', 'value', allow_duplicate=True),
                       Output('add-mask-boundary', 'value', allow_duplicate=True),
                       Output('add-cell-id-mask-hover', 'value', allow_duplicate=True),
                       Output('main-tabs', 'active_tab', allow_duplicate=True),
                       prevent_initial_call=True)
    def update_parameters_from_config_json_or_db(uploaded_w_data, new_blend_dict, db_config_selection, data_selection,
                                                 current_blend_dict, error_config, db_config_list, cur_metadata, delimiter):
        """
        Update the blend layer dictionary and currently selected channels from a JSON-formatted upload
        Only applies to the channels that have already been selected: if channels are not in the current blend,
        they will be modified on future selection
        Requires that the channel modification menu be empty to make sure that parameters are updated properly
        """
        if ctx.triggered_id == "db-config-options" and db_config_selection is not None:
            new_blend_dict = match_db_config_to_request_str(db_config_list, db_config_selection)
        metadata_return = extract_alias_labels_from_db_document(new_blend_dict, cur_metadata)
        metadata_return = metadata_return if len(metadata_return) > 0 else dash.no_update
        if None not in (uploaded_w_data, new_blend_dict, data_selection):
            # reformat the blend dict to remove the metadata key if reported with h5py so it will match
            current_blend_dict = {key: value for key, value in current_blend_dict.items() if 'metadata' not in key}
            if panel_match(current_blend_dict, new_blend_dict) or all_roi_match(
                    current_blend_dict, new_blend_dict, uploaded_w_data, delimiter):
                current_blend_dict = new_blend_dict['channels'].copy()
                all_layers = {data_selection: {}}
                error_config = add_warning_to_error_config(error_config, ALERT.warnings["json_update_success"])
                channel_list_return = dash.no_update
                if 'config' in new_blend_dict and 'blend' in new_blend_dict['config'] and \
                        all([elem in current_blend_dict.keys() for elem in new_blend_dict['config']['blend']]):
                    channel_list_return = new_blend_dict['config']['blend']
                    for elem in channel_list_return:
                        current_blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                            current_blend_dict, elem, uploaded_w_data, data_selection)
                        array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem], current_blend_dict[elem])
                        all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                        current_blend_dict[elem]['color'])).astype(np.uint8)
                global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma = \
                    parse_global_filter_values_from_json(new_blend_dict['config'])
                clust_return = {data_selection: new_blend_dict['cluster']} if \
                    'cluster' in new_blend_dict.keys() and new_blend_dict['cluster'] else dash.no_update
                gate_return = new_blend_dict['gating'] if 'gating' in new_blend_dict.keys() else dash.no_update
                apply, level, boundary, hover = mask_options_from_json(new_blend_dict)
                return SessionServerside(all_layers, key="layer_dict", use_unique_key=OVERWRITE), \
                    current_blend_dict, error_config, channel_list_return, global_apply_filter, global_filter_type, \
                    global_filter_val, global_filter_sigma, metadata_return, dash.no_update, \
                    dash.no_update, clust_return, gate_return, None, apply, level, boundary, hover, "pixel-analysis"
            # IMP: if the update does not occur, clear the database selection and autofill config name
            return no_json_db_updates(add_warning_to_error_config(error_config, ALERT.warnings["json_update_error"]))
        elif data_selection is None:
            return no_json_db_updates(add_warning_to_error_config(error_config, ALERT.warnings["json_requires_roi"]))
        raise PreventUpdate

    @dash_app.callback(Input('image_layers', 'value'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('param_config', 'data'),
                       State('canvas-layers', 'data'),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'),
                       State('images_in_blend', 'value'),
                       State('autofill-channel-colors', 'value'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Output('param_config', 'data', allow_duplicate=True),
                       Output('images_in_blend', 'value', allow_duplicate=True),
                       prevent_initial_call=True)
    def update_blend_dict_on_channel_selection(add_to_layer, uploaded_w_data, current_blend_dict, data_selection,
                                               param_dict, all_layers, preset_selection, preset_dict,
                                               cur_image_in_mod_menu, autofill_channel_colours):
        """
        Update the blend dictionary when a new channel is added to the multichannel selector
        """
        if None not in (add_to_layer, current_blend_dict, data_selection, uploaded_w_data) and data_selection in uploaded_w_data:
            try:
                channel_modify = dash.no_update
                if param_dict is None or len(param_dict) < 1: param_dict = {"current_roi": data_selection}
                if data_selection is not None:
                    if current_blend_dict is not None and "current_roi" in param_dict.keys() and \
                            data_selection != param_dict["current_roi"]:
                        param_dict["current_roi"] = data_selection
                        if cur_image_in_mod_menu is not None and cur_image_in_mod_menu in current_blend_dict.keys():
                            channel_modify = cur_image_in_mod_menu
                    else:
                        param_dict["current_roi"] = data_selection
                all_layers = check_empty_missing_layer_dict(all_layers, data_selection)
                for elem in add_to_layer:
                    # if the selected channel doesn't have a config yet, create one either from scratch or a preset
                    if elem not in current_blend_dict.keys() and not preset_selection and uploaded_w_data:
                        current_blend_dict[elem] = {'color': '#FFFFFF', 'x_lower_bound': 0, 'x_upper_bound':
                            get_default_channel_upper_bound_by_percentile(uploaded_w_data[data_selection][elem]),
                                                    'filter_type': None, 'filter_val': None, 'filter_sigma': None}
                        if autofill_channel_colours:
                            current_blend_dict = select_random_colour_for_channel(current_blend_dict, elem, DEFAULT_COLOURS)
                        if None not in (preset_selection, preset_dict):
                            current_blend_dict[elem] = apply_preset_to_blend_dict(current_blend_dict[elem], preset_dict[preset_selection])
                    # if the selected channel is in the current blend, check if a preset is used to override
                    elif elem in current_blend_dict.keys() and None not in (preset_selection, preset_dict):
                        # do not override the colour of the current channel
                        current_blend_dict[elem] = apply_preset_to_blend_dict(current_blend_dict[elem], preset_dict[preset_selection])
                    else:
                        if autofill_channel_colours:
                            current_blend_dict = select_random_colour_for_channel(current_blend_dict, elem, DEFAULT_COLOURS)
                        current_blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                            current_blend_dict, elem, uploaded_w_data, data_selection)
                    if data_selection in all_layers.keys() and (
                            elem not in all_layers[data_selection].keys() or preset_selection):
                        array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem], current_blend_dict[elem])
                        all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                        current_blend_dict[elem]['color'])).astype(np.uint8)
                return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                                                             use_unique_key=OVERWRITE), param_dict, channel_modify
            except (TypeError, KeyError): raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output("annotation-color-picker", 'value', allow_duplicate=True),
                       Output('swatch-color-picker', 'value'),
                       Input('swatch-color-picker', 'value'),
                       prevent_initial_call=True)
    def update_colour_picker_from_swatch(swatch):
        # IMP: need to reset the value of the swatch to None after transferring the colour
        if swatch is not None: return dict(hex=swatch), None
        raise PreventUpdate

    @dash_app.callback(Input("annotation-color-picker", 'value'),
                       State('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('image_layers', 'value'),
                       State('canvas-layers', 'data'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       State('bool-apply-filter', 'value'),
                       State('filter-type', 'value'),
                       State("kernel-val-filter", 'value'),
                       State("sigma-val-filter", 'value'),
                       State('images_in_blend', 'options'),
                       prevent_initial_call=True)
    def update_blend_dict_on_color_selection(colour, layer, uploaded_w_data,
                                             current_blend_dict, data_selection, add_to_layer,
                                             all_layers, filter_chosen, filter_name, filter_value, filter_sigma,
                                             blend_options):
        """
        Update the blend dictionary and layer dictionary when a modification channel changes its colour
        """
        if None not in (layer, current_blend_dict, data_selection):
            array = uploaded_w_data[data_selection][layer]
            if current_blend_dict[layer]['color'] != colour['hex']:
                blend_options = [elem['value'] for elem in blend_options]
                if all([elem in add_to_layer for elem in blend_options]):
                    # if upper and lower bounds have been set before for this layer, use them before recolouring
                    if current_blend_dict[layer]['x_lower_bound'] is not None and \
                            current_blend_dict[layer]['x_upper_bound'] is not None:
                        array = filter_by_upper_and_lower_bound(array, float(current_blend_dict[layer]['x_lower_bound']),
                                                                float(current_blend_dict[layer]['x_upper_bound']))
                    array = apply_filter_to_channel(array, filter_chosen, filter_name, filter_value, filter_sigma)
                    current_blend_dict[layer]['color'] = colour['hex']
                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array, colour['hex'])).astype(np.uint8)
                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict", use_unique_key=OVERWRITE)
                raise PreventUpdate
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(State('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('canvas-layers', 'data'),
                       Input('pixel-intensity-slider', 'value'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       prevent_initial_call=True)
    def set_blend_params_on_pixel_range_adjustment(layer, uploaded_w_data, current_blend_dict, data_selection,
                                                   all_layers, slider_values):

        if None not in (slider_values, layer, data_selection, uploaded_w_data) and \
                all([elem is not None for elem in slider_values]):
            # do not update if the range values in the slider match the current blend params:
            try:
                slider_values = [float(elem) for elem in slider_values]
                lower_bound, upper_bound = min(slider_values), max(slider_values)
                if float(current_blend_dict[layer]['x_lower_bound']) == float(lower_bound) and \
                        float(current_blend_dict[layer]['x_upper_bound']) == float(upper_bound): raise PreventUpdate
                else:
                    current_blend_dict[layer]['x_lower_bound'] = float(lower_bound)
                    current_blend_dict[layer]['x_upper_bound'] = float(upper_bound)
                    array = apply_preset_to_array(uploaded_w_data[data_selection][layer], current_blend_dict[layer])
                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array, current_blend_dict[layer]['color']))
                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict", use_unique_key=OVERWRITE)
            except (TypeError, KeyError): raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(State('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('canvas-layers', 'data'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'),
                       prevent_initial_call=True)
    def set_blend_params_on_preset_selection(layer, uploaded_w_data, current_blend_dict, data_selection, all_layers,
                                             preset_selection, preset_dict):
        """
        Set the blend param dictionary and canvas layer dictionary when a preset is applied to the current ROI.
        """
        if None not in (preset_selection, preset_dict, data_selection, current_blend_dict, layer):
            for preset_val in ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val', 'filter_sigma']:
                current_blend_dict[layer][preset_val] = preset_dict[preset_selection][preset_val]
            array = apply_preset_to_array(uploaded_w_data[data_selection][layer], preset_dict[preset_selection])
            all_layers[data_selection][layer] = np.array(recolour_greyscale(array, current_blend_dict[layer]['color']))
            return current_blend_dict, SessionServerside(all_layers, key="layer_dict", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @dash_app.callback(State('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('canvas-layers', 'data'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Input('bool-apply-filter', 'value'),
                       Input('filter-type', 'value'),
                       Input("kernel-val-filter", 'value'),
                       Input("sigma-val-filter", 'value'),
                       State('image_layers', 'value'),
                       State('images_in_blend', 'options'),
                       State('static-session-var', 'data'),
                       prevent_initial_call=True)
    def set_blend_options_for_layer_with_bool_filter(layer, uploaded, current_blend_dict, data_selection,
                                                     all_layers, filter_chosen, filter_name, filter_value, filter_sigma,
                                                     cur_layers, blend_options, session_vars):
        only_options_changed = False
        if None not in (ctx.triggered, session_vars):
            only_options_changed = channel_already_added(ctx.triggered_id, ctx.triggered, session_vars)
        if None not in (layer, current_blend_dict, data_selection, filter_value, filter_name, all_layers,
                        filter_sigma) and not only_options_changed:
            try:
                array = uploaded[data_selection][layer]
            except KeyError: array = None
            # condition where the current inputs are set to not have a filter, and the current blend dict matches
            no_filter_in_both = no_filter_chosen(current_blend_dict, layer, filter_chosen)
            # condition where toggling between two channels, and the first one has no filter and the second
            # has a filter. prevent the callback with no actual change
            same_filter_params = channel_filter_matches(current_blend_dict, layer, filter_chosen,
                                                        filter_name, filter_value, filter_sigma)

            # do not update if the gaussian filter is applied with an even number
            gaussian_even = filter_name == "gaussian" and (int(filter_value) % 2 == 0)

            if not no_filter_in_both and not same_filter_params and not gaussian_even and array is not None:
                # do not update if all of the channels are not in the Channel dict
                blend_options = [elem['value'] for elem in blend_options]
                if all([elem in cur_layers for elem in blend_options]):

                    if current_blend_dict[layer]['x_lower_bound'] is not None and \
                            current_blend_dict[layer]['x_upper_bound'] is not None:
                        array = filter_by_upper_and_lower_bound(array, float(current_blend_dict[layer]['x_lower_bound']),
                                                                float(current_blend_dict[layer]['x_upper_bound']))

                    if len(filter_chosen) > 0 and filter_name is not None:
                        array = apply_filter_to_channel(array, filter_chosen, filter_name, filter_value, filter_sigma)
                        current_blend_dict = set_blend_parameters_for_channel(current_blend_dict, layer,
                                                                              filter_name, filter_value, filter_sigma)

                    else:
                        current_blend_dict = set_blend_parameters_for_channel(current_blend_dict, layer,
                                            filter_name, filter_value, filter_sigma, clear=True)

                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                        current_blend_dict[layer]['color'])).astype(np.uint8)
                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict", use_unique_key=OVERWRITE)
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Input('data-collection', 'value'),
        Output('canvas-layers', 'data', allow_duplicate=True))
    def reset_canvas_layers_on_new_dataset(data_selection):
        """
        Reset the canvas layers dictionary containing the cached images for the current canvas in order to
        retain memory. Should be cleared on a new ROI selection if caching is not retained
        If caching is enabled, then the blended arrays that form the image will be retained for quicker
        toggling
        """
        return {data_selection: {}} if data_selection else dash.no_update

    @dash_app.callback(Output('blending_colours', 'data', allow_duplicate=True),
                       Input('preset-options', 'value'),
                       Input('image_presets', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       prevent_initial_call=True)
    def set_blend_options_from_preset(preset_selection, preset_dict, current_blend_dict, data_selection):
        if None not in (preset_selection, preset_dict, current_blend_dict, data_selection):
            for key, value in current_blend_dict.items():
                current_blend_dict[key] = apply_preset_to_blend_dict(value, preset_dict[preset_selection])
            return current_blend_dict
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Output('mask-options', 'value', allow_duplicate=True),
                       Input('data-collection', 'value'),
                       State('data-collection', 'options'),
                       State('mask-options', 'options'),
                       State('annotation_canvas', 'figure'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    def clear_canvas_and_set_mask_on_new_dataset(new_selection, dataset_options, mask_options, cur_canvas, delimiter):
        """
        Reset the canvas to blank on an ROI change
        Will attempt to set the new mask based on the ROI name and the list of mask options
        """
        if new_selection:
            return CanvasLayout(cur_canvas).get_fig(), ROIMaskMatch(new_selection, mask_options,
                                                        dataset_options, delimiter, True).get_match()
        raise PreventUpdate

    @dash_app.callback(
        Output('channel-order', 'data'),
        Input('set-sort', 'n_clicks'),
        State('blend-options-ag-grid', 'virtualRowData'),
        State('channel-order', 'data'),
        Input('image_layers', 'value'),
        State('alias-dict', 'data'),
        prevent_initial_call=True)
    def set_channel_sorting(nclicks, rowdata, channel_order, current_blend, aliases):
        """
        Set the channel order in a dcc Store based on the dash ag grid or adding/removing a channel from the list
        """
        return set_channel_list_order(nclicks, rowdata, channel_order, current_blend, aliases, ctx.triggered_id)

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Output('download-canvas-image-tiff', 'data'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('status-div', 'children'),
                       Input('canvas-layers', 'data'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input('alias-dict', 'data'),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       State('uploaded_dict', 'data'),
                       Input('channel-intensity-hover', 'value'),
                       State('param_config', 'data'),
                       State('mask-dict', 'data'),
                       Input('apply-mask', 'value'),
                       Input('mask-options', 'value'),
                       State('toggle-canvas-legend', 'value'),
                       State('toggle-canvas-scalebar', 'value'),
                       Input('mask-blending-slider', 'value'),
                       Input('add-mask-boundary', 'value'),
                       State('channel-order', 'data'),
                       State('legend-size-slider', 'value'),
                       Input('add-cell-id-mask-hover', 'value'),
                       State('pixel-size-ratio', 'value'),
                       State('invert-annotations', 'value'),
                       Input('overlay-grid-canvas', 'value'),
                       State('legend_orientation', 'value'),
                       Input('bool-apply-global-filter', 'value'),
                       Input('global-filter-type', 'value'),
                       Input("global-kernel-val-filter", 'value'),
                       Input("global-sigma-val-filter", 'value'),
                       Input('toggle-cluster-annotations', 'value'),
                       Input('cluster-colour-assignments-dict', 'data'),
                       Input('cluster-col', 'value'),
                       State('imported-cluster-frame', 'data'),
                       Input('cluster-annotation-type', 'value'),
                       Input('btn-download-canvas-tiff', 'n_clicks'),
                       State('custom-scale-val', 'value'),
                       State('cluster-annotations-legend', 'value'),
                       Input('apply-gating', 'value'),
                       Input('gating-cell-list', 'data'),
                       State('dataset-delimiter', 'value'),
                       State('scalebar-color', 'value'),
                       State('session_alert_config', 'data'),
                       Input('cluster-label-selection', 'value'),
                       State('canvas-div-holder', 'children'),
                       prevent_initial_call=True)
    # @time_taken_callback
    def render_canvas_from_layer_mask_hover_change(canvas_layers, currently_selected,
                                                   data_selection, blend_colour_dict, aliases, cur_graph,
                                                   cur_graph_layout, raw_data_dict,
                                                   show_each_channel_intensity, param_dict, mask_config, mask_toggle,
                                                   mask_selection, toggle_legend, toggle_scalebar, mask_blending_level,
                                                   add_mask_boundary,
                                                   channel_order, legend_size, add_cell_id_hover, pixel_ratio,
                                                   invert_annot, overlay_grid, legend_orientation,
                                                   global_apply_filter, global_filter_type, global_filter_val,
                                                   global_filter_sigma,
                                                   apply_cluster_on_mask, cluster_assignments_dict, cluster_cat,
                                                   cluster_frame, cluster_type,
                                                   download_canvas_tiff, custom_scale_val,
                                                   cluster_assignments_in_legend, apply_gating, gating_cell_id_list,
                                                   delimiter, scale_color, error_config, clust_selected, canvas_holder):

        """
        Update the canvas from either an underlying change to the source image, or a change to the hover template
        Examples of triggers include:
        - changes to the additive blend (channel modifications & global filters)
        - toggling of a mask and its related features (blending, clustering)
        - if the hover template is updated (it is faster to recreate the figure rather than trying to remove the
        hover template)
        """
        # do not update if the trigger is a global filter and the filter is not applied
        global_not_enabled = global_filter_disabled(ctx.triggered_id, global_apply_filter)
        channel_order_same = channel_order_as_default(ctx.triggered_id, channel_order, currently_selected)
        # gating always triggers an update, so prevent this here
        dont_update = ctx.triggered_id == "gating-cell-list" and gating_cell_id_list is None
        empty_mask = no_canvas_mask(ctx.triggered_id, mask_selection, mask_toggle)
        if layers_exist(canvas_layers, data_selection) and currently_selected and blend_colour_dict and data_selection \
                and len(channel_order) > 0 and not global_not_enabled and not channel_order_same and canvas_holder and \
                data_selection in canvas_layers and canvas_layers[data_selection] and not dont_update and not empty_mask:
            cur_graph = strip_invalid_shapes_from_graph_layout(cur_graph)
            legend_text = generate_canvas_legend_text(blend_colour_dict, channel_order, aliases, legend_orientation,
                        cluster_assignments_in_legend, cluster_assignments_dict, data_selection, clust_selected, cluster_cat)
            try:
                canvas = CanvasImage(canvas_layers, data_selection, currently_selected, mask_config, mask_selection,
                mask_blending_level, overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover, show_each_channel_intensity,
                raw_data_dict, aliases, global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma,
                apply_cluster_on_mask, cluster_assignments_dict, cluster_cat, cluster_frame, cluster_type,
                custom_scale_val, apply_gating, gating_cell_id_list, scale_color, clust_selected)
                fig = canvas.generate_canvas()
                if cluster_type == 'mask' or not apply_cluster_on_mask:
                    fig = CanvasLayout(fig).remove_cluster_annotation_shapes()
                elif apply_cluster_on_mask and cluster_cat:
                    fig = CanvasLayout(fig).add_cluster_annotations_as_circles(mask_config[mask_selection]["raw"],
                        pd.DataFrame(cluster_frame[data_selection]), cluster_assignments_dict, data_selection, 2,
                        apply_gating, gating_cell_id_list, clust_selected, cluster_cat)
                # set if the image is to be downloaded or not
                dest_path = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
                canvas_tiff, download_status = dash.no_update, dash.no_update
                if ctx.triggered_id == "btn-download-canvas-tiff":
                    fig = dash.no_update
                    canvas_tiff = dcc.send_file(output_current_canvas_as_tiff(canvas_image=canvas.get_image(),
                                dest_dir=dest_path, use_roi_name=True, roi_name=data_selection, delimiter=delimiter))
                    download_status = timestamp_download_child()
                return (fig.to_dict() if isinstance(fig, go.Figure) else fig), canvas_tiff, dash.no_update, download_status
            except Exception as e:
                error_config = add_warning_to_error_config(error_config, str(e))
                return reset_graph_with_malformed_template(cur_graph), dash.no_update, error_config, dash.no_update
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Output('annotation_canvas', 'relayoutData', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       Input('annotation_canvas', 'relayoutData'),
                       State('set-x-auto-bound', 'value'),
                       State('set-y-auto-bound', 'value'),
                       State('window_config', 'data'),
                       Input('activate-coord', 'n_clicks'),
                       State('data-collection', 'value'),
                       State('uploaded_dict', 'data'),
                       Input('custom-scale-val', 'value'),
                       Input('pixel-size-ratio', 'value'),
                       State('toggle-canvas-scalebar', 'value'),
                       State('legend-size-slider', 'value'),
                       State('invert-annotations', 'value'),
                       State('scalebar-color', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_coord_or_zoom_change(cur_graph, cur_graph_layout, x_request, y_request, current_window,
                                                nclicks_coord, data_selection, image_dict, custom_scale_val,
                                                pixel_ratio, toggle_scalebar, legend_size, invert_annot, scale_col):
        """
        Update the annotation canvas when the zoom or custom coordinates are requested.
        """
        # update the scale bar with and without zooming
        if None not in (cur_graph, cur_graph_layout, data_selection):
            cur_graph = strip_invalid_shapes_from_graph_layout(cur_graph)
            if ctx.triggered_id not in ["activate-coord"]:
                try:
                    image_shape = get_first_image_from_roi_dictionary(image_dict[data_selection]).shape
                    proportion = float(custom_scale_val / image_shape[1]) if custom_scale_val is not None else 0.1
                    cur_graph = CanvasLayout(cur_graph).update_scalebar_zoom_value(cur_graph_layout, pixel_ratio, proportion, scale_col)
                    x_axis_placement = set_x_axis_placement_of_scalebar(image_shape[1], invert_annot)
                    cur_graph = CanvasLayout(cur_graph).toggle_scalebar(toggle_scalebar, x_axis_placement, invert_annot,
                                pixel_ratio, image_shape, legend_size, proportion, scale_col)
                    return cur_graph, cur_graph_layout
                except (ValueError, KeyError, AssertionError): raise PreventUpdate
            if ctx.triggered_id == "activate-coord":
                if None not in (x_request, y_request, current_window) and \
                        nclicks_coord is not None and nclicks_coord > 0:
                    try:
                        fig, new_layout = CanvasLayout(cur_graph).update_coordinate_window(current_window, x_request, y_request)
                        return fig, new_layout
                    except (AssertionError, TypeError): raise PreventUpdate
                raise PreventUpdate
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Input('toggle-canvas-legend', 'value'),
                       Input('toggle-canvas-scalebar', 'value'),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       State('alias-dict', 'data'),
                       State('uploaded_dict', 'data'),
                       Input('channel-order', 'data'),
                       State('legend-size-slider', 'value'),
                       State('pixel-size-ratio', 'value'),
                       State('invert-annotations', 'value'),
                       Input('legend_orientation', 'value'),
                       State('custom-scale-val', 'value'),
                       Input('cluster-annotations-legend', 'value'),
                       State('cluster-colour-assignments-dict', 'data'),
                       State('cluster-col', 'value'),
                       Input('scalebar-color', 'value'),
                       State('cluster-label-selection', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_toggle_show_annotations(toggle_legend, toggle_scalebar,
                                                   cur_canvas, cur_layout, currently_selected,
                                                   data_selection, blend_colour_dict, aliases, image_dict,
                                                   channel_order, legend_size, pixel_ratio, invert_annot,
                                                   legend_orientation, custom_scale_val, cluster_assignments_in_legend,
                                                   cluster_assignments_dict, cluster_cat, scalebar_col, clust_selected):
        """
        re-render the canvas if the user requests to remove the annotations (scalebar and legend) or
        updates the scalebar length with a custom value
        """
        # do not trigger update if the channel order is maintained
        chan_same = channel_order_as_default(ctx.triggered_id, channel_order, currently_selected)
        if None not in (cur_layout, cur_canvas, data_selection, currently_selected, blend_colour_dict) and not chan_same:
            image_shape = get_first_image_from_roi_dictionary(image_dict[data_selection]).shape
            x_axis_placement = set_x_axis_placement_of_scalebar(image_shape[1], invert_annot)
            cur_canvas = CanvasLayout(cur_canvas).clear_improper_shapes()
            if ctx.triggered_id in ["toggle-canvas-legend", "legend_orientation", "cluster-annotations-legend", "channel-order"]:
                legend_text = generate_canvas_legend_text(blend_colour_dict, channel_order, aliases, legend_orientation,
                            cluster_assignments_in_legend, cluster_assignments_dict,
                            data_selection, clust_selected, cluster_cat) if toggle_legend else ''
                canvas = CanvasLayout(cur_canvas).toggle_legend(toggle_legend, legend_text, x_axis_placement, legend_size)
                return CanvasLayout(canvas).clear_improper_shapes()
            elif ctx.triggered_id in ["toggle-canvas-scalebar", "scalebar-color"]:
                proportion = float(custom_scale_val / image_shape[1]) if custom_scale_val is not None else 0.1
                canvas = CanvasLayout(cur_canvas).toggle_scalebar(toggle_scalebar, x_axis_placement, invert_annot,
                        pixel_ratio, image_shape, legend_size, proportion, scalebar_col)
                return CanvasLayout(canvas).get_fig()
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input('invert-annotations', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_invert_annotations(cur_canvas, cur_layout, currently_selected,
                                              data_selection, blend_colour_dict, invert_annotations):
        if None not in (cur_layout, cur_canvas, data_selection, currently_selected, blend_colour_dict):
            return invert_annotations_figure(strip_invalid_shapes_from_graph_layout(cur_canvas))
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       Input('legend-size-slider', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_change_annotation_size(cur_graph, legend_size):
        """
        Update the canvas when the size of the annotations is modified
        """
        if cur_graph is not None:
            try: return CanvasLayout(cur_graph).change_annotation_size(legend_size)
            except KeyError: raise PreventUpdate
        raise PreventUpdate

    @du.callback(Output('panel_config', 'data'),
                 id='upload-panel-info')
    def upload_custom_panel_info(status: du.UploadStatus):
        """
        Upload a metadata panel separate from the auto-generated metadata panel. This must be parsed against the existing
        datasets to ensure that it matches the number of channels
        """
        files = DashUploaderFileReader(status).return_filenames()
        if files:
            return {'uploads': list(files)}
        raise PreventUpdate

    @dash_app.callback(
        Output("imc-panel-editable", "columns", allow_duplicate=True),
        Output("imc-panel-editable", "data", allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Input('panel_config', 'data'),
        State('uploaded_dict_template', 'data'),
        State('session_alert_config', 'data'),
        State('imc-panel-editable', 'data'),
        prevent_initial_call=True)
    def populate_datatable_columns(panel_config, uploaded, error_config, cur_metadata):
        if panel_config is not None and len(panel_config['uploads']) > 0:
            metadata_read = pd.read_csv(panel_config['uploads'][0])
            metadata_validated = validate_incoming_metadata_table(metadata_read, uploaded)
            if metadata_validated is not None:
                # make sure that the internal keys from channel names stay the same
                metadata_validated['Channel Name'] = pd.DataFrame(cur_metadata)['Channel Name']
                if 'rakaia Label' not in metadata_validated.keys():
                    metadata_validated['rakaia Label'] = metadata_validated["Channel Label"]
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                        p in metadata_validated.keys()], pd.DataFrame(metadata_validated).to_dict(orient='records'), dash.no_update
            error_config = add_warning_to_error_config(error_config, ALERT.warnings["custom_metadata_error"])
            return dash.no_update, dash.no_update, error_config
        raise PreventUpdate

    @dash_app.callback(
        Output("imc-panel-editable", "columns"),
        Output("imc-panel-editable", "data"),
        Input('uploaded_dict_template', 'data'))
    def populate_metadata_table(uploaded):
        if uploaded is not None and uploaded['metadata'] is not None:
            try:
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                        p in uploaded['metadata'].keys()], pd.DataFrame(uploaded['metadata']).to_dict(orient='records')
            except ValueError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Input("imc-panel-editable", "data"),
        Output('alias-dict', 'data'))
    def create_channel_label_dict(metadata):
        if metadata is not None:
            return populate_alias_dict_from_editable_metadata(metadata)

    @dash_app.callback(
        Output("download-edited-table", "data"),
        Input("btn-download-panel", "n_clicks"),
        Input("imc-panel-editable", "data"))
    def download_edited_metadata(n_clicks, datatable_contents):
        if n_clicks and datatable_contents is not None and ctx.triggered_id == "btn-download-panel":
            return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "panel.csv", index=False)
        raise PreventUpdate

    @dash_app.callback(Output('download-canvas-image-html', 'data'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('btn-download-canvas-html', 'n_clicks'),
                       State('annotation_canvas', 'figure'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('annotation_canvas', 'style'),
                       State('data-collection', 'value'),
                       State('dataset-delimiter', 'value'),
                       State('session_alert_config', 'data'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def download_interactive_html_canvas(download_html, cur_graph, uploaded, blend_dict, canvas_style,
                                         dataset_selection, delimiter, error_config):
        if None not in (cur_graph, uploaded, blend_dict) and download_html:
            try:
                html_path = dcc.send_file(output_current_canvas_as_html(download_html, cur_graph, canvas_style,
                            use_roi_name=True, roi_name=dataset_selection, delimiter=delimiter))
                error_config = dash.no_update
            except Exception as e:
                error_config = add_warning_to_error_config(error_config, str(e))
                html_path = dash.no_update
            return html_path, error_config
        raise PreventUpdate

    @dash_app.callback(Output('download-session-config-json', 'data'),
                       Input('btn-download-session-config-json', 'n_clicks'),
                       State('blending_colours', 'data'),
                       State('image_layers', 'value'),
                       State('bool-apply-global-filter', 'value'),
                       State('global-filter-type', 'value'),
                       State("global-kernel-val-filter", 'value'),
                       State("global-sigma-val-filter", 'value'),
                       State('cluster-colour-assignments-dict', 'data'),
                       State('data-collection', 'value'),
                       State('alias-dict', 'data'),
                       State('gating-dict', 'data'),
                       State('apply-mask', 'value'),
                       State('mask-blending-slider', 'value'),
                       State('add-mask-boundary', 'value'),
                       State('add-cell-id-mask-hover', 'value'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def download_session_config_json(download_json, blend_dict, blend_layers, global_apply_filter,
                                     global_filter_type, global_filter_val, global_filter_sigma, cluster_assignments,
                                     data_selection, aliases, gating_dict, apply_mask, mask_level, mask_boundary,
                                     mask_hover):
        if blend_dict and download_json: return dcc.send_file(write_blend_config_to_json(download_json, blend_dict,
        blend_layers, global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma, data_selection,
        cluster_assignments, aliases, gating_dict, apply_mask, mask_level, mask_boundary, mask_hover))
        raise PreventUpdate

    @dash_app.callback(Output('download-roi-h5py', 'data'),
                       Input('btn-download-roi-h5py', 'n_clicks'),
                       State('uploaded_dict', 'data'),
                       State('imc-panel-editable', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('annotation_canvas', 'relayoutData'),
                       State('graph-subset-download', 'value'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def update_download_href_h5(download_h5py, uploaded, metadata_sheet, blend_dict, data_selection,
                                canvas_layout, graph_subset):
        """
        Create the download links for the current canvas and the session data.
        Only update if the download dialog is open to avoid continuous updating on canvas change
        """
        if None not in (uploaded, blend_dict) and download_h5py:
            first_image = get_first_image_from_roi_dictionary(uploaded[data_selection])
            try:
                mask = None
                if 'shapes' in canvas_layout and ' use graph subset on download' in graph_subset:
                    mask = subset_mask_for_data_export(canvas_layout, first_image.shape)
                return dcc.send_file(write_session_data_to_h5py(download_h5py, metadata_sheet,
                uploaded, data_selection, blend_dict, mask))
            # if the dictionary hasn't updated to include all the experiments, then don't update download just yet
            except KeyError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output('annotation_canvas', 'style'),
        Input('annotation-canvas-size', 'value'),
        State('annotation_canvas', 'figure'),
        State('data-collection', 'value'),
        State('annotation_canvas', 'style'),
        Input('cur_roi_dimensions', 'data'),
        Input('canvas-viewport-max', 'value'),
        prevent_initial_call=False)
    def update_canvas_size(value, current_canvas, data_selection, cur_sizing, cur_dimensions, viewport_max):
        if None not in (value, data_selection, cur_dimensions, viewport_max):
            return set_canvas_viewport(value, None, data_selection, current_canvas, cur_sizing, cur_dimensions, viewport_max)
        return {'width': f'{value}vh', 'height': f'{value}vh'}

    @dash_app.callback(
        Output("selected-area-table", "data"),
        Input('annotation_canvas', 'figure'),
        Input('annotation_canvas', 'relayoutData'),
        State('uploaded_dict', 'data'),
        State('image_layers', 'value'),
        State('data-collection', 'value'),
        State('alias-dict', 'data'),
        Input("compute-region-statistics", "n_clicks"),
        Input("area-stats-collapse", "is_open"),
        prevent_initial_call=True)
    def update_area_information(graph, graph_layout, upload, layers, data_selection, aliases_dict, nclicks,
                                stats_table_open):
        if None not in (graph, graph_layout, data_selection) and stats_table_open:
            return RegionSummary(graph, graph_layout, upload, layers, data_selection, aliases_dict).get_summary_frame()
        elif stats_table_open: return generate_empty_region_table()
        raise PreventUpdate

    @dash_app.callback(Output('image-gallery-row', 'children'),
                       Input('uploaded_dict', 'data'),
                       Input('data-collection', 'value'),
                       State('annotation_canvas', 'relayoutData'),
                       Input('toggle-gallery-zoom', 'value'),
                       State('preset-options', 'value'),
                       State('image_presets', 'data'),
                       Input('toggle-gallery-view', 'value'),
                       Input('unique-channel-list', 'value'),
                       Input('alias-dict', 'data'),
                       State('preset-button', 'n_clicks'),
                       State('blending_colours', 'data'),
                       Input('default-scaling-gallery', 'value'),
                       State('session_config', 'data'),
                       State('dataset-delimiter', 'value'),
                       State('data-collection', 'options'),
                       State('image-gallery-row', 'children'),
                       Input('chan-gallery-zoom-update', 'n_clicks'),
                       prevent_initial_call=True)
    def create_channel_tile_gallery_grid(gallery_data, data_selection, canvas_layout, toggle_gallery_zoom,
                                         preset_selection, preset_dict, view_by_channel, channel_selected, aliases,
                                         nclicks, blend_colour_dict, toggle_scaling_gallery, session_config, delimiter,
                                         options, cur_gal, update_zoom):
        """
        Create a tiled image gallery of the current ROI. If the current dataset selection does not yet have
        default percentile scaling applied, apply before rendering
        IMPORTANT: do not return the blend dictionary here as it will override the session blend on an ROI change
        """
        try:
            # do not update if the canvas triggers, but gallery zoom is not enabled
            zoom_not_needed = ctx.triggered_id == 'chan-gallery-zoom-update' and not toggle_gallery_zoom
            data_there = data_selection in gallery_data.keys() and \
                         all([elem is not None for elem in gallery_data[data_selection].values()])
            # 1. if a channel is selected, but view by channel is not enabled
            # 2. if view by channel is enabled but no channel is selected
            no_channel = ctx.triggered_id == "unique-channel-list" and not view_by_channel or \
                         (ctx.triggered_id == "toggle-gallery-view" and not channel_selected)
            # don't use updated aliases if using single-channel view
            dont_need_aliases = ctx.triggered_id == "alias-dict" and (view_by_channel and channel_selected)
            if data_there and not zoom_not_needed and not no_channel and not dont_need_aliases:
                # if the aliases are changed but the gallery exists, just update the labels in the DOM without re-rendering
                if ctx.triggered_id == "alias-dict" and cur_gal and len(cur_gal) == len(aliases):
                    return replace_channel_gallery_aliases(cur_gal, aliases)
                else:
                    # maintain the original order of channels that is dictated by the metadata
                    # decide if channel view or ROI view is selected
                    if view_by_channel and channel_selected:
                        views = RegionThumbnail(session_config, blend_colour_dict, [channel_selected], 1000000,
                        delimiter=delimiter, use_greyscale=True, dataset_options=options, single_channel_view=True).get_image_dict()
                        if toggle_scaling_gallery:
                            try:
                                blend_colour_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                                    blend_colour_dict, channel_selected, gallery_data, data_selection)
                                views = {key: apply_preset_to_array(value, blend_colour_dict[channel_selected]) for
                                         key, value in views.items()}
                            except KeyError: pass
                    else:
                        views = {elem: gallery_data[data_selection][elem] for elem in list(aliases.keys())}
                    toggle_gallery_zoom = toggle_gallery_zoom if not view_by_channel else False
                    return generate_channel_tile_gallery_children(views, canvas_layout, ZOOM_KEYS,
                    blend_colour_dict, preset_selection, preset_dict, aliases, nclicks, toggle_gallery_zoom,
                    toggle_scaling_gallery, 0.75, 3000, channel_selected if (view_by_channel and
                                                channel_selected) else None) if views else []
            raise PreventUpdate
        except (dash.exceptions.LongCallbackError, AttributeError, KeyError):
            raise PreventUpdate

    @dash_app.server.route("/" + str(tmpdirname) + "/" + str(authentic_id) + '/downloads/<path:path>')
    def serve_static(path):
        return flask.send_from_directory(os.path.join(tmpdirname, str(authentic_id), 'downloads'), path, as_attachment=True)

    @dash_app.callback(Output('blend-options-ag-grid', 'rowData'),
                       Output('blend-options-ag-grid', 'defaultColDef'),
                       Input('blending_colours', 'data'),
                       Input('channel-order', 'data'),
                       State('data-collection', 'value'),
                       Input('alias-dict', 'data'))
    def create_ag_grid_legend(blend_colours, current_blend, data_selection, aliases):
        """
        Set the inputs and parameters for the dash ag grid containing the current blend channels
        """
        if current_blend is not None and len(current_blend) > 0:
            in_blend = [aliases[elem] for elem in current_blend]
            cell_styling_conditions = ag_grid_cell_styling_conditions(blend_colours, current_blend, data_selection, aliases)
            if len(in_blend) > 0 and len(cell_styling_conditions) > 0:
                to_return = pd.DataFrame(in_blend, columns=["Channel"]).to_dict(orient="records")
                return to_return, {"sortable": False, "filter": False, "cellStyle": {"styleConditions": cell_styling_conditions}}
            return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), {"sortable": False, "filter": False}
        return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), {"sortable": False, "filter": False}

    @dash_app.callback(
        Output("area-stats-collapse", "is_open", allow_duplicate=True),
        [Input("compute-region-statistics", "n_clicks")],
        [State("area-stats-collapse", "is_open")])
    def toggle_area_stats_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("extra-import-collapse", "is_open", allow_duplicate=True),
        [Input("show-collapse-extra-imports", "n_clicks")],
        [State("extra-import-collapse", "is_open")])
    def toggle_extra_imports_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(Output("pixel-hist", 'figure', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'value', allow_duplicate=True),
                       Input('data-collection', 'value'),
                       State('images_in_blend', 'value'),
                       prevent_initial_call=True)
    def reset_pixel_adjustments_on_new_dataset(new_selection, currently_in_blend):
        """
        Reset the pixel histogram and range slider on a new dataset selection
        """
        if currently_in_blend is not None: return reset_pixel_histogram(), [None, None]
        raise PreventUpdate

    @dash_app.callback(Output("pixel-hist", 'figure', allow_duplicate=True),
                       Output("annotation_canvas", 'figure', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'value', allow_duplicate=True),
                       Output('images_in_blend', 'options', allow_duplicate=True),
                       Output('images_in_blend', 'value', allow_duplicate=True),
                       State('images_in_blend', 'options'),
                       Input('image_layers', 'value'),
                       State("annotation_canvas", 'figure'),
                       prevent_initial_call=True)
    def reset_graphs_on_empty_modification_menu(current_selection, blend, cur_canvas):
        """
        reset all the relevant input widgets and dropdown menus when there is no channel currently selected
        """
        if blend is None or len(blend) == 0 and len(current_selection) > 0 and cur_canvas:
            cur_canvas['data'] = []
            return reset_pixel_histogram(), cur_canvas, [None, None], [], None
        raise PreventUpdate

    @dash_app.callback(Input('images_in_blend', 'value'),
                       Output('custom-slider-max', 'value'),
                       prevent_initial_call=True)
    def reset_range_max_on_channel_switch(new_image_mod):
        """
        Reset the checkbox for a custom range slider max on channel changing. Prevents the slider bar from
        having incorrect bounds for the upcoming channel
        """
        return [] if new_image_mod else dash.no_update

    @dash_app.callback(Output("pixel-hist", 'figure'),
                       Output('pixel-intensity-slider', 'max'),
                       Output('pixel-intensity-slider', 'value'),
                       Output('pixel-intensity-slider', 'marks'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'step'),
                       Output("pixel-hist-collapse", "is_open", allow_duplicate=True),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       Input('blending_colours', 'data'),
                       Input("pixel-hist-collapse", "is_open"),
                       State('pixel-intensity-slider', 'value'),
                       Input('custom-slider-max', 'value'),
                       prevent_initial_call=True)
    def update_pixel_histogram_and_intensity_sliders(selected_channel, uploaded, data_selection,
                                                     current_blend_dict, show_pixel_hist, cur_slider_values,
                                                     custom_max):
        """
        Create pixel histogram and output the default percentiles
        """
        # TODO: currently, the pixel histogram will collapse on a slider change because of the blend dictionary.
        # collapse is triggered by this object to prevent the pixel histogram from being empty on an ROI change
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict):
            blend_return, hist_open = dash.no_update, dash.no_update
            try:
                if show_pixel_hist and ctx.triggered_id in ["pixel-hist-collapse", "images_in_blend"]:
                    fig, hist_max = pixel_hist_from_array(uploaded[data_selection][selected_channel])
                else:
                    fig, hist_open = dash.no_update, False
                    hist_max = float(np.max(uploaded[data_selection][selected_channel]))
            except (ValueError, TypeError, KeyError):
                fig, hist_max, hist_open = dash.no_update, 100.0, False
            try:
                tick_markers, step_size = set_range_slider_tick_markers(hist_max)
            except ValueError:
                hist_max = 100.0
                tick_markers, step_size = set_range_slider_tick_markers(hist_max)
            # if the hist is triggered by the changing of a channel to modify or a new blend dict
            # set the min of the hist max to be 1 for very low images to also match the min for the pixel hist max
            hist_max = float(hist_max if hist_max > 1 else 1)
            if ctx.triggered_id in ["images_in_blend"]:
                try:
                    # if the current selection has already had a histogram bound on it, update the histogram with it
                    if current_blend_dict[selected_channel]['x_lower_bound'] is not None and \
                            current_blend_dict[selected_channel]['x_upper_bound'] is not None:
                        lower_bound = float(current_blend_dict[selected_channel]['x_lower_bound'])
                        upper_bound = float(current_blend_dict[selected_channel]['x_upper_bound'])
                    else:
                        lower_bound = 0
                        upper_bound = get_default_channel_upper_bound_by_percentile(uploaded[data_selection][selected_channel])
                        current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                        current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                        blend_return = current_blend_dict
                    # if the upper bound is larger than the custom percentile, set it to the upper bound
                    if ' Set range max to current upper bound' in custom_max:
                        hist_max = upper_bound
                        tick_markers, step_size = set_range_slider_tick_markers(hist_max)
                    # set tick spacing between marks on the rangeslider
                    # have 4 tick markers
                    return fig, hist_max, [lower_bound, upper_bound], tick_markers, blend_return, step_size, hist_open
                except (KeyError, ValueError):
                    return {}, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False
            elif ctx.triggered_id == 'blending_colours':
                vals_return = dash.no_update
                if current_blend_dict[selected_channel]['x_lower_bound'] is not None and \
                        current_blend_dict[selected_channel]['x_upper_bound'] is not None:
                    if (cur_slider_values[0] is None or cur_slider_values[1] is None) or (
                            float(current_blend_dict[selected_channel]['x_lower_bound']) != float(cur_slider_values[0])
                            or float(current_blend_dict[selected_channel]['x_upper_bound']) != float(cur_slider_values[1])):
                        lower_bound = float(current_blend_dict[selected_channel]['x_lower_bound'])
                        upper_bound = float(current_blend_dict[selected_channel]['x_upper_bound'])
                        vals_return = [lower_bound, upper_bound]
                else:
                    lower_bound = 0
                    upper_bound = get_default_channel_upper_bound_by_percentile(uploaded[data_selection][selected_channel])
                    current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                    current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                    blend_return = current_blend_dict
                    vals_return = [lower_bound, upper_bound]
                hist_max = hist_max if not custom_max else dash.no_update
                tick_markers = tick_markers if not custom_max else dash.no_update
                return dash.no_update, hist_max, vals_return, tick_markers, blend_return, step_size, hist_open
            elif ctx.triggered_id == "pixel-hist-collapse":
                return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, hist_open
            elif ctx.triggered_id == 'custom-slider-max':
                try:
                    if ' Set range max to current upper bound' in custom_max:
                        if current_blend_dict[selected_channel]['x_upper_bound'] >= cur_slider_values[1]:
                            hist_max = float(cur_slider_values[1])
                        else:
                            hist_max = upper_bound_for_range_slider(uploaded[data_selection][selected_channel])
                    else:
                        # if the toggle is reset, make sure it works properly for values below 1
                        hist_max = upper_bound_for_range_slider(uploaded[data_selection][selected_channel])
                    tick_markers, step_size = set_range_slider_tick_markers(hist_max)
                    return dash.no_update, hist_max, cur_slider_values, tick_markers, dash.no_update, step_size, hist_open
                except IndexError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('pixel-intensity-slider', 'max', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'value', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'marks', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'step', allow_duplicate=True),
                       Output('custom-slider-max', 'value', allow_duplicate=True),
                       State('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       State('pixel-intensity-slider', 'value'),
                       Input('set-default-rangeslider', 'n_clicks'),
                       State('custom-slider-max', 'value'),
                       prevent_initial_call=True)
    def reset_intensity_slider_to_default(selected_channel, uploaded, data_selection, current_blend_dict,
                                          cur_slider_values, reset, cur_max):
        """
        Reset the range slider for the current channel to the default values (min of 0 and max of 99th pixel
        percentile)
        """
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict):
            hist_max = upper_bound_for_range_slider(uploaded[data_selection][selected_channel])
            upper_bound = float(get_default_channel_upper_bound_by_percentile(uploaded[data_selection][selected_channel]))
            if int(cur_slider_values[0]) != 0 or (int(cur_slider_values[1]) != upper_bound):
                tick_markers, step_size = set_range_slider_tick_markers(hist_max)
                return hist_max, [0, upper_bound], tick_markers, step_size, []
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('bool-apply-filter', 'value'),
                       Output('filter-type', 'value'),
                       Output('kernel-val-filter', 'value'),
                       Output('sigma-val-filter', 'value'),
                       Output("annotation-color-picker", 'value'),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       Input('blending_colours', 'data'),
                       Input('preset-options', 'value'),
                       State('image_presets', 'data'),
                       State('static-session-var', 'data'),
                       State('bool-apply-filter', 'value'),
                       State('filter-type', 'value'),
                       State('kernel-val-filter', 'value'),
                       State('sigma-val-filter', 'value'),
                       State("annotation-color-picker", 'value'),
                       State('autofill-channel-colors', 'value'))
    def update_channel_filter_inputs(selected_channel, uploaded, data_selection, current_blend_dict,
                                     preset_selection, preset_dict, session_vars, cur_bool_filter, cur_filter_type,
                                     cur_filter_val, cur_filter_sigma, cur_colour, col_autofill):
        """
        Update the input widgets wth the correct channel configs when the channel is changed, or a preset is used,
        or if the blend dict is updated
        """
        only_options_changed = False
        if None not in (ctx.triggered, session_vars):
            only_options_changed = channel_already_added(ctx.triggered_id, ctx.triggered, session_vars)
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict) and \
                ctx.triggered_id in ["images_in_blend", "blending_colours"] and not only_options_changed:
            filter_type, filter_val, filter_sigma, color = return_current_channel_blend_params(current_blend_dict, selected_channel)
            to_apply_filter = return_current_or_default_filter_apply(cur_bool_filter, filter_type, filter_val, filter_sigma)
            filter_type_return = return_current_or_default_filter_param(cur_filter_type, filter_type)
            filter_val_return = return_current_or_default_filter_param(cur_filter_val, filter_val)
            filter_sigma_return = return_current_or_default_filter_param(cur_filter_sigma, filter_sigma)
            color_return = return_current_or_default_channel_color(cur_colour, color, col_autofill)
            return to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return
        if ctx.triggered_id in ['preset-options'] and None not in \
                (preset_selection, preset_dict, selected_channel, data_selection, current_blend_dict):
            filter_type, filter_val, filter_sigma, color = return_current_channel_blend_params(current_blend_dict, selected_channel)
            to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return = \
                return_current_default_params_with_preset(filter_type, filter_val, filter_sigma, color)
            return to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return
        raise PreventUpdate

    @dash_app.callback(Output('sigma-val-filter', 'disabled'),
                       Input('filter-type', 'value'),
                       prevent_initial_call=True)
    def update_channel_filter_type(filter_type):
        return True if filter_type == "median" else False

    @dash_app.callback(Output('global-sigma-val-filter', 'disabled'),
                       Input('global-filter-type', 'value'),
                       prevent_initial_call=True)
    def update_global_channel_filter_inputs(filter_type):
        return True if filter_type == "median" else False

    @dash_app.callback(Input('preset-button', 'n_clicks'),
                       State('set-preset', 'value'),
                       State('preset-options', 'options'),
                       State('data-collection', 'value'),
                       State('images_in_blend', 'value'),
                       State('blending_colours', 'data'),
                       State('image_presets', 'data'),
                       State('preset-options', 'value'),
                       Output('preset-options', 'options'),
                       Output('image_presets', 'data'),
                       Output('preset-options', 'value'))
    def generate_preset_options(selected_click, preset_name, current_preset_options, data_selection, layer,
                                current_blend_dict, current_presets, cur_preset_chosen):
        if selected_click and None not in (preset_name, data_selection, layer, current_blend_dict):
            if preset_name not in current_preset_options:
                current_preset_options.append(preset_name)
            current_presets = {} if current_presets is None else current_presets
            current_presets[preset_name] = current_blend_dict[layer]
            set_preset = cur_preset_chosen if cur_preset_chosen in current_preset_options else None
            return current_preset_options, current_presets, set_preset
        raise PreventUpdate

    @dash_app.callback(Input('image_presets', 'data'),
                       Output('hover-preset-information', 'children'))
    def update_hover_preset_information(preset_dict):
        """
        Update the hover information on the list of presets so that the user can preview the parameters before selecting
        """
        if preset_dict:
            text = generate_preset_options_preview_text(preset_dict)
            return html.Textarea(text, style={"width": "200px", "height": f"{100 * len(preset_dict)}px"})
        raise PreventUpdate

    @dash_app.callback(Input('session_config', 'data'),
                       Output('unique-channel-list', 'options'),
                       Input('alias-dict', 'data'),
                       prevent_initial_call=True)
    def populate_gallery_channel_list(session_config, aliases):
        """
        Populate a list of all unique channel names for the gallery view
        """
        if session_config is not None and 'unique_images' in session_config.keys():
            try:
                if not all([elem in aliases.keys() for elem in session_config['unique_images']]): raise AssertionError
                return [{'label': aliases[i], 'value': i} for i in session_config['unique_images']]
            except AttributeError: raise DataImportError(ALERT.warnings["possible-disk-storage-error"])
            except KeyError: return []
        return []

    @dash_app.callback(Output('static-session-var', 'data'),
                       Input('images_in_blend', 'value'),
                       State('static-session-var', 'data'),
                       prevent_initial_call=True)
    def save_cur_channel_in_memory(selected_channel, cur_vars):
        """
        Keep track of the current channel selected for modification to avoid extraneous callbacks
        """
        if selected_channel is not None:
            cur_vars = {} if cur_vars is None else cur_vars
            try: cur_vars["cur_channel"] = selected_channel
            except KeyError: pass
            return cur_vars
        raise PreventUpdate

    @dash_app.callback(
        Output("fullscreen-canvas", "is_open"),
        Input('make-canvas-fullscreen', 'n_clicks'),
        [State("fullscreen-canvas", "is_open")])
    def toggle_fullscreen_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(Output('annotation_canvas-fullscreen', 'figure'),
                       Output('annotation_canvas-fullscreen', 'relayoutData'),
                       Input('make-canvas-fullscreen', 'n_clicks'),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas-fullscreen', 'relayoutData'),
                       prevent_initial_call=True)
    def render_canvas_fullscreen(clicks, cur_canvas, cur_layout):
        if clicks > 0 and None not in (cur_layout, cur_canvas):
            fig = FullScreenCanvas(cur_canvas, cur_layout)
            return fig.get_canvas(True), fig.get_canvas_layout()
        return {}, None

    @dash_app.callback(State('annotation_canvas', 'figure'),
                       Input('annotation_canvas', 'relayoutData'),
                       Output('bound-shower', 'children'),
                       Output('window_config', 'data'),
                       prevent_initial_call=True)
    def update_bound_display(cur_graph, cur_graph_layout):
        bound_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
        if None not in (cur_graph, cur_graph_layout) and all([elem in cur_graph_layout for elem in bound_keys]):
            # only update if these keys are used for drag or pan to set custom coordinates
            return bounds_text(*high_low_values_from_zoom_layout(cur_graph_layout))
        # if the zoom is reset to the default, clear the bound window
        elif cur_graph_layout in [{'xaxis.autorange': True, 'yaxis.autorange': True}, {'autosize': True}]:
            return [], {"x_low": None, "x_high": None, "y_low": None, "y_high": None}
        raise PreventUpdate

    @dash_app.callback(
        Output("dataset-preview", "is_open"),
        Input('show-dataset-info', 'n_clicks'),
        [State("dataset-preview", "is_open")])
    def toggle_dataset_info_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("data-collection", "value", allow_duplicate=True),
        Input('dataset-preview-table', 'selected_rows'),
        State('data-collection', 'options'),
        State('data-collection', 'value'),
        prevent_initial_call=True)
    def select_roi_from_preview_table(active_selection, dataset_options, cur_selection):
        if None not in (active_selection, dataset_options) and len(active_selection) > 0:
            try:
                return dataset_options[active_selection[0]] if \
                    dataset_options[active_selection[0]] != cur_selection else dash.no_update
            except KeyError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output('dataset-preview-table', 'selected_rows'),
        Input("data-collection", "value"),
        State('data-collection', 'options'),
        prevent_initial_call=True)
    def update_selected_preview_row_on_roi_selection(data_selection, dataset_options):
        if None not in (dataset_options, data_selection):
            try:
                return [dataset_options.index(data_selection)]
            except KeyError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output("alert-modal", "is_open"),
        Output("alert-information", "children"),
        Input('session_alert_config', 'data'),
        State('toggle-session-messages', 'value'),
        prevent_initial_call=True)
    def show_alert_modal(alert_dict, show_messages):
        """
        If the alert dict is populated with a warning, show the warning in the modal. Otherwise, do not populate and
        don't show the modal
        """
        if alert_dict and "error" in alert_dict.keys() and alert_dict["error"] is not None and show_messages:
            return True, [html.H6("Message: \n"), html.H6(alert_dict["error"])]
        return False, None

    @dash_app.callback(Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('alias-dict', 'data'),
                       Input("imc-panel-editable", "data"),
                       State('session_alert_config', 'data'),
                       prevent_initial_call=True)
    def give_alert_on_improper_edited_metadata(gene_aliases, metadata_editable, error_config):
        """
        Send an alert when the format of the editable metadata table looks incorrect
        Will arise if more labels are provided than there are channels, which will create blank key entries
        in the metadata list and alias dictionary
        """
        if any([elem in ['', ' '] for elem in gene_aliases.keys()]) or any([elem['Channel Name'] in
            ['', ' '] or elem['Channel Label'] in ['', ' '] for elem in metadata_editable]):
            return add_warning_to_error_config(error_config, ALERT.warnings["metadata_format_error"])
        raise PreventUpdate

    @dash_app.callback(Output('region-annotation', 'disabled'),
                       Input('annotation_canvas', 'relayoutData'),
                       State('data-collection', 'value'),
                       Input('image_layers', 'value'),
                       prevent_initial_call=True)
    def enable_region_annotation_on_layout(cur_graph_layout, data_selection, current_blend):
        """
        Enable the region annotation button to be selectable when the canvas is either zoomed in on, or
        a shape is being added/edited. These represent a region selection that can be annotated
        """
        if None not in (cur_graph_layout, data_selection, current_blend) and len(current_blend) > 0:
            if all([elem in cur_graph_layout for elem in ZOOM_KEYS]) or 'shapes' in cur_graph_layout and \
                    len(cur_graph_layout['shapes']) > 0: return False
            return True
        return True

    @dash_app.callback(
        Output("region-annotation-modal", "is_open"),
        Input('region-annotation', 'n_clicks'),
        Input('create-annotation', 'n_clicks'))
    def toggle_region_annotation_modal(clicks_add_annotation, clicks_submit_annotation):
        if clicks_add_annotation and ctx.triggered_id == "region-annotation": return True
        elif ctx.triggered_id == "create-annotation" and clicks_submit_annotation: return False
        return False

    @dash_app.callback(
        Output("annotations-dict", "data"),
        Input('create-annotation', 'n_clicks'),
        State('region-annotation-name', 'value'),
        State('region-annotation-body', 'value'),
        State('region-annotation-cell-types', 'value'),
        State('annotation_canvas', 'relayoutData'),
        State('annotations-dict', 'data'),
        State('data-collection', 'value'),
        State('image_layers', 'value'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        State('mask-blending-slider', 'value'),
        State('add-mask-boundary', 'value'),
        State('quant-annotation-col', 'value'),
        Input('gating-annotation-create', 'n_clicks'),
        State('apply-gating', 'value'),
        State('quant-annotation-col-gating', 'value'),
        State('gating-annotation-assignment', 'value'),
        State('gating-cell-list', 'data'),
        State('bulk-annotate-shapes', 'value'))
    def add_annotation_to_dict(create_annotation, annotation_title, annotation_body, annotation_cell_type,
                               canvas_layout, annotations_dict, data_selection, cur_layers, mask_toggle,
                               mask_selection, mask_blending_level, add_mask_boundary, annot_col, add_annot_gating,
                               apply_gating, gating_annot_col, gating_annot_type, gating_cell_id_list, bulk_annot):
        annotations_dict = check_for_valid_annotation_hash(annotations_dict, data_selection)
        # Option 1: if triggered from gating
        if ctx.triggered_id == "gating-annotation-create" and add_annot_gating and apply_gating and None not in \
                (gating_annot_col, gating_annot_type, gating_cell_id_list, mask_selection, data_selection, cur_layers):
            annotations_dict[data_selection][tuple(gating_cell_id_list)] = RegionAnnotation(title=None, body=None,
                cell_type=gating_annot_type, imported=False, annotation_column=gating_annot_col, type="gate",
                channels=cur_layers, use_mask=mask_toggle, mask_selection=mask_selection,
                mask_blending_level=mask_blending_level, add_mask_boundary=add_mask_boundary, id=str(shortuuid.uuid())).dict()
            return SessionServerside(annotations_dict, key="annotation_dict", use_unique_key=OVERWRITE)
        # Option 2: if triggered from region drawing
        elif ctx.triggered_id == "create-annotation" and create_annotation and None not in \
                (annotation_title, annotation_body, canvas_layout, data_selection, cur_layers):
            annotation_list = generate_annotation_list(canvas_layout, bulk_annot)
            for key, value in annotation_list.items():
                annotations_dict[data_selection][key] = RegionAnnotation(title=annotation_title, body=annotation_body,
                cell_type=annotation_cell_type, imported=False, annotation_column=annot_col, type=value,
                channels=cur_layers, use_mask=mask_toggle, mask_selection=mask_selection,
                mask_blending_level=mask_blending_level, add_mask_boundary=add_mask_boundary, id=str(shortuuid.uuid())).dict()
            return SessionServerside(annotations_dict, key="annotation_dict", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @dash_app.callback(Output('annotation-table', 'data'),
                       Output('annotation-table', 'columns'),
                       Input("annotations-dict", "data"),
                       Input('data-collection', 'value'),
                       prevent_initial_call=True)
    def populate_annotations_table_preview(annotations_dict, dataset_selection):
        if None not in (annotations_dict, dataset_selection):
            return annotation_preview_table(annotations_dict, dataset_selection)
        raise PreventUpdate

    @dash_app.callback(
        Output("inputs-offcanvas", "is_open"),
        Input("inputs-offcanvas-button", "n_clicks"),
        State("inputs-offcanvas", "is_open"))
    def toggle_offcanvas_inputs(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("blend-config-offcanvas", "is_open"),
        Input("blend-offcanvas-button", "n_clicks"),
        State("blend-config-offcanvas", "is_open"))
    def toggle_offcanvas_blend_options(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("color-picker-collapse", "is_open", allow_duplicate=True),
        [Input("show-color-picker", "n_clicks")],
        [State("color-picker-collapse", "is_open")])
    def toggle_color_picker_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("pixel-hist-collapse", "is_open", allow_duplicate=True),
        [Input("show-pixel-hist", "n_clicks")],
        [State("pixel-hist-collapse", "is_open")])
    def toggle_pixel_hist_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("channel-list-collapse", "is_open", allow_duplicate=True),
        [Input("show-channel-collapse", "n_clicks")],
        [State("channel-list-collapse", "is_open")])
    def toggle_draggable_list_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("annotation-preview", "is_open"),
        Input('show-annotation-table', 'n_clicks'),
        [State("annotation-preview", "is_open")])
    def toggle_annotation_table_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("quant-annotation-col", "options"),
        Output('quant-annotation-col-in-tab', 'options'),
        Output('quant-annotation-col-gating', 'options'),
        Output('annotation-cat-click', 'options'),
        Input('add-annotation-col', 'n_clicks'),
        State('new-annotation-col', 'value'),
        State('quant-annotation-col', 'options'),
        Input('add-annot-col-quantification', 'n_clicks'),
        State('annotation-col-quantification', 'value'),
        State('quant-annotation-col-in-tab', 'options'),
        prevent_initial_call=True)
    def add_new_annotation_column(add_new_col_canvas, new_col_canvas, current_cols_canvas,
                                  add_new_col_quant, new_col_quant, current_cols_quant):
        """
        Add a new annotation column to the dropdown menu possibilities for annotation
        Will add the category to all the dropdown menus where annotation can occur: Currently these are:
            - region (under Region/presets -> Add region annotation)
            - gating (under Configuration -> Gating)
            - quantification (Under Quantification/clustering -> UMAP options)
        All dropdown outputs will always have the same columns
        """
        # Cases where the callback will occur: if either button is clicked and the corresponding input field is not empty
        trigger_canvas = ctx.triggered_id == "add-annotation-col" and add_new_col_canvas > 0 and new_col_canvas
        trigger_quant = ctx.triggered_id == "add-annot-col-quantification" and add_new_col_quant > 0 and new_col_quant
        if trigger_canvas or trigger_quant:
            if not set(current_cols_canvas) == set(current_cols_quant): raise AssertionError
            col_to_add = new_col_canvas if ctx.triggered_id == "add-annotation-col" else new_col_quant
            cur_cols = current_cols_canvas.copy()
            if not isinstance(cur_cols, list) and len(cur_cols) > 0: raise AssertionError
            if col_to_add not in cur_cols: cur_cols.append(col_to_add)
            return cur_cols, cur_cols, cur_cols, cur_cols
        raise PreventUpdate

    @dash_app.callback(
        Output("annotations-dict", "data", allow_duplicate=True),
        Output('click-annotation-alert', 'children'),
        Output('click-annotation-alert', 'is_open'),
        Output('annotation_canvas', 'figure', allow_duplicate=True),
        Input('annotation_canvas', 'clickData'),
        State('click-annotation-assignment', 'value'),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        State('annotation-cat-click', 'value'),
        State('annotation_canvas', 'figure'),
        State('enable_click_annotation', 'value'),
        State('click-annotation-add-circle', 'value'),
        State('annotation-circle-size', 'value'),
        prevent_initial_call=True)
    def add_annotation_to_dict_with_click(clickdata, annotation_cell_type, annotations_dict,
                                          data_selection, annot_col, cur_figure, enable_click_annotation,
                                          add_circle, circle_size):

        if None not in (clickdata, data_selection, cur_figure) and enable_click_annotation and 'points' in clickdata:
            try:
                annotations_dict = check_for_valid_annotation_hash(annotations_dict, data_selection)
                x, y = clickdata['points'][0]['x'], clickdata['points'][0]['y']
                annotations_dict[data_selection][str(clickdata)] = RegionAnnotation(title=None, body=None, cell_type=
                annotation_cell_type, imported=False, annotation_column=annot_col, type='point', channels=None,
                use_mask=False, mask_selection=None, mask_blending_level=None, add_mask_boundary=False, id=str(shortuuid.uuid())).dict()
                fig = dash.no_update if not add_circle else CanvasLayout(cur_figure).add_click_point_circle(x, y, circle_size)
                return SessionServerside(annotations_dict, key="annotation_dict"), \
                    html.H6(f"Point {x, y} updated with {annotation_cell_type} in {annot_col}"), True, fig
            except KeyError:
                return dash.no_update, html.H6("Error in annotating point"), True, dash.no_update
        raise PreventUpdate

    @dash_app.callback(
        Output('annotation_canvas', 'figure', allow_duplicate=True),
        Input('imported-annotations-csv', 'data'),
        State('uploaded_dict', 'data'),
        State('data-collection', 'value'),
        State('annotation_canvas', 'figure'),
        State('annotation-circle-size', 'value'),
        prevent_initial_call=True)
    def populate_canvas_with_point_annotation_circles(imported_annotations, image_dict, data_selection,
                                                      cur_graph, circle_size):
        """
        Render a circle for every valid point annotation imported from a CSV. Valid xy coordinates
        must fit inside the dimensions of the current image
        """
        if None not in (imported_annotations, image_dict, data_selection, cur_graph):
            first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
            fig = CanvasLayout(cur_graph).add_point_annotations_as_circles(imported_annotations, first_image, circle_size)
            return CanvasLayout(fig).clear_improper_shapes()
        raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'value', allow_duplicate=True),
                       Output('apply-mask', 'value', allow_duplicate=True),
                       Input('prev-roi', 'n_clicks'),
                       Input('next-roi', 'n_clicks'),
                       Input('keyboard-listener', 'event'),
                       Input('keyboard-listener', 'n_events'),
                       State('data-collection', 'value'),
                       State('data-collection', 'options'),
                       State('enable-roi-change-key', 'value'),
                       State('region-annotation-modal', 'is_open'),
                       State('main-tabs', 'active_tab'),
                       State('tour_component', 'isOpen'),
                       State('apply-mask', 'value'),
                       prevent_initial_call=True)
    def use_key_listener(prev_roi, next_roi, key_listener, n_events, cur_data_selection, cur_options,
                         allow_arrow_change, annotating_region, active_tab, open_tour, mask_stat):
        """
        Use the key event listener to trigger the following actions:
            - Use the forward and backwards buttons to click to a new ROI
            - Alternatively, use the directional arrow buttons from an event listener
            - Use the arrow up button to toggle on/off the mask
        """
        if None not in (cur_data_selection, cur_options) and not (ctx.triggered_id == 'keyboard-listener' and not allow_arrow_change) and \
            not annotating_region and active_tab == 'pixel-analysis' and not open_tour and valid_key_trigger(key_listener):
            cur_index = cur_options.index(cur_data_selection)
            mask_change = not mask_stat if mask_toggle_trigger(ctx.triggered_id, key_listener, n_events) else dash.no_update
            try:
                prev_trigger = previous_roi_trigger(ctx.triggered_id, prev_roi, key_listener, n_events)
                next_trigger = next_roi_trigger(ctx.triggered_id, next_roi, key_listener, n_events)
                if prev_trigger and cur_index != 0:
                    return cur_options[cur_index - 1] if cur_options[cur_index - 1] != cur_data_selection else dash.no_update, mask_change
                elif next_trigger:
                    return cur_options[cur_index + 1] if cur_options[cur_index + 1] != cur_data_selection else dash.no_update, mask_change
                else: return dash.no_update, mask_change
            except IndexError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('prev-roi', 'disabled'),
                       Output('next-roi', 'disabled'),
                       Input('data-collection', 'value'),
                       State('data-collection', 'options'),
                       prevent_initial_call=True)
    def toggle_roi_click_through_visibility(cur_data_selection, cur_options):
        """
        Toggle the visibility/availability of the previous and next ROI buttons depending on the current ROI selection
        """
        disabled_prev = True if cur_options and cur_options[0] == cur_data_selection else False
        disabled_next = True if cur_options and cur_options[-1] == cur_data_selection else False
        return disabled_prev, disabled_next

    @dash_app.callback(
        Output('image_layers', 'value', allow_duplicate=True),
        Output('main-tabs', 'active_tab'),
        Input({'type': 'gallery-channel', "index": ALL}, "n_clicks"),
        State('image_layers', 'options'),
        State('image_layers', 'value'),
        State('alias-dict', 'data'),
        State('main-tabs', 'active_tab'),
        prevent_initial_call=True)
    def add_channel_layer_through_gallery_click(value, layer_options, current_blend, aliases, active_tab):
        """
        Add a channel from the channel thumbnail gallery with component pattern matching.
        Ensure that the gallery tab is active for a switch to occur.
        """
        if not all([elem is None for elem in value]) and None not in (layer_options, current_blend, aliases) and \
                active_tab == 'gallery-tab':
            index_from = ctx.triggered_id["index"]
            if index_from in [i["value"] for i in layer_options] and index_from not in current_blend:
                current_blend.append(index_from)
                return current_blend, "pixel-analysis"
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output("session-config-modal", "is_open"),
        Input('session-config-modal-button', 'n_clicks'),
        [State("session-config-modal", "is_open")])
    def toggle_general_config_modal(n, is_open):
        """
        Open the modal for general session variables
        """
        return not is_open if n else is_open

    @dash_app.callback(Output('annotation_canvas', 'config', allow_duplicate=True),
                       Input('enable-canvas-scroll-zoom', 'value'),
                       State('annotation_canvas', 'config'),
                       prevent_initial_call=True)
    def toggle_scroll_zoom_on_canvas(enable_zoom, cur_config):
        """
        Toggle the ability to use scroll zoom on the annotation canvas using the input from the
        session configuration modal. Default value is not enabled
        """
        if 'scrollZoom' in cur_config: cur_config['scrollZoom'] = enable_zoom
        return cur_config

    @dash_app.callback(Output('tour_component', 'isOpen'),
                       Input('dash-import-tour', 'n_clicks'),
                       prevent_initial_call=True)
    def open_tour_guide(activate_tour):
        """
        Toggle open the import tour if requested
        """
        return True if activate_tour else False

    @dash_app.callback(Output('marker-cor-display', 'children'),
                       Input('target-channel-cor', 'value'),
                       Input('baseline-channel-cor', 'value'),
                       State('uploaded_dict', 'data'),
                       Input('data-collection', 'value'),
                       State('mask-dict', 'data'),
                       Input('apply-mask', 'value'),
                       Input('mask-options', 'value'),
                       Input('blending_colours', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       prevent_initial_call=True)
    def show_marker_correlation(target, baseline, image_dict, roi_selection, mask_dict, apply_mask, mask_selection,
                                blending_dict, bounds):
        """
        Display the marker correlation statistics for a target and baseline (if provided)
        """
        if target and image_dict and roi_selection:
            mask = mask_dict[mask_selection]["raw"] if (mask_dict and mask_selection and apply_mask) else None
            target_mask, prop, target_baseline, pearson = MarkerCorrelation(image_dict, roi_selection, target, baseline,
                                    mask=mask, blend_dict=blending_dict, bounds=bounds).get_correlation_statistics()
            return marker_correlation_children(target_mask, prop, target_baseline, pearson)
        return []

    @dash_app.callback(Output('saved-blends', 'data'),
                       Output('saved-blend-options', 'options'),
                       Output('saved-blend-options-roi', 'options'),
                       Input('save-cur-blend', 'n_clicks'),
                       State('name-cur-blend', 'value'),
                       State('saved-blends', 'data'),
                       State('saved-blend-options', 'options'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       prevent_initial_call=True)
    def create_saved_blend(save_blend, blend_name, saved_blend_dict,
                           saved_blend_options, cur_selected_channels, data_selection):
        """
        Generate a saved blend from the current canvas as a named entity. Will save the current channels and
        blend parameters in a name configuration for rapid loading or ROI querying
        """
        if None not in (blend_name, cur_selected_channels, data_selection):
            if blend_name not in saved_blend_options: saved_blend_options.append(blend_name)
            return add_saved_blend(saved_blend_dict, blend_name, cur_selected_channels), saved_blend_options, saved_blend_options
        raise PreventUpdate

    @dash_app.callback(Output('image_layers', 'value', allow_duplicate=True),
                       Input('saved-blend-options', 'value'),
                       State('blending_colours', 'data'),
                       State('saved-blends', 'data'),
                       prevent_initial_call=True)
    def load_saved_blend(blend_chosen, blend_params, saved_blend_dict):
        """
        Load a saved blend by name. This will override the current channels in the canvas and replace
        them with the saved blend
        """
        if blend_chosen and blend_params and saved_blend_dict and blend_chosen in saved_blend_dict:
            return [i for i in saved_blend_dict[blend_chosen]]
        raise PreventUpdate

    @dash_app.callback(Output('blend-config-offcanvas', 'placement'),
                       Input('toggle-advanced-placement', 'value'),
                       prevent_initial_call=False)
    def toggle_advanced_settings_placement(toggle_placement):
        """
        Change the page position of the advanced tools sidebar (masking, gating, clustering, etc.)
        """
        return "start" if toggle_placement else "end"
