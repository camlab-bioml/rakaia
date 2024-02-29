import os.path
import dash.exceptions
import dash_uploader as du
import flask
from dash import ctx, ALL
from dash_extensions.enrich import Output, Input, State, html
from ccramic.inputs.pixel_level_inputs import (
    wrap_canvas_in_loading_screen_for_large_images,
    invert_annotations_figure,
    set_range_slider_tick_markers,
    generate_canvas_legend_text,
    set_x_axis_placement_of_scalebar, update_canvas_filename)
from ccramic.parsers.pixel_level_parsers import (
    FileParser,
    populate_image_dict_from_lazy_load,
    create_new_blending_dict,
    populate_alias_dict_from_editable_metadata,
    check_blend_dictionary_for_blank_bounds_by_channel)
from ccramic.utils.pixel_level_utils import (
    delete_dataset_option_from_list_interactively,
    split_string_at_pattern,
    get_default_channel_upper_bound_by_percentile,
    apply_preset_to_array,
    recolour_greyscale,
    resize_for_canvas,
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
    ag_grid_cell_styling_conditions)
# from ccramic.utils.session import remove_ccramic_caches
from ccramic.components.canvas import CanvasImage, CanvasLayout
from ccramic.io.display import (
    RegionSummary,
    output_current_canvas_as_tiff,
    output_current_canvas_as_html,
    FullScreenCanvas,
    generate_preset_options_preview_text,
    annotation_preview_table)
from ccramic.io.gallery_outputs import generate_channel_tile_gallery_children
from ccramic.parsers.cell_level_parsers import match_mask_name_with_roi
from ccramic.utils.graph_utils import strip_invalid_shapes_from_graph_layout
from ccramic.inputs.loaders import (
    previous_roi_trigger,
    next_roi_trigger,
    adjust_option_height_from_list_length)
from ccramic.callbacks.pixel_level_wrappers import parse_global_filter_values_from_json
from ccramic.io.session import (
    write_blend_config_to_json,
    write_session_data_to_h5py,
    subset_mask_for_data_export,
    create_download_dir,
    SessionServerside)
# from ccramic.parsers.cell_level_parsers import validate_coordinate_set_for_image
from pathlib import Path
from plotly.graph_objs.layout import YAxis, XAxis
import json
import cv2
from dash import dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import median_filter
from natsort import natsorted
from ccramic.io.readers import DashUploaderFileReader
from ccramic.utils.db import (
    match_db_config_to_request_str,
    extract_alias_labels_from_db_document)
from ccramic.utils.alert import AlertMessage, file_import_message, DataImportError
import uuid
from ccramic.utils.region import (
    RegionAnnotation,
    check_for_valid_annotation_hash)
from ccramic.parsers.roi_parsers import RegionThumbnail
from ccramic.utils.filter import (
    return_current_or_default_filter_apply,
    return_current_or_default_filter_param,
    return_current_channel_blend_params,
    return_current_or_default_channel_color,
    return_current_default_params_with_preset)
import shortuuid

def init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with pixel level analysis/raw image preprocessing (image loading,
    blending, filtering, scaling, etc.)
    """
    dash_app.config.suppress_callback_exceptions = True
    DEFAULT_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#00FAFF", "#FF00FF", "#FFFF00", "#FFFFFF"]
    ALERT = AlertMessage()

    @du.callback(Output('uploads', 'data'),
                 id='upload-image')
    # @cache.memoize())
    def get_filenames_from_drag_and_drop(status: du.UploadStatus):
        uploader = DashUploaderFileReader(status)
        files = uploader.return_filenames()
        if files is not None:
            return files
        raise PreventUpdate

    @du.callback(Output('param_blend_config', 'data', allow_duplicate=True),
                 id='upload-param-json')
    # @cache.memoize())
    def get_param_json_from_drag_and_drop(status: du.UploadStatus):
        uploader = DashUploaderFileReader(status)
        files = uploader.return_filenames()
        if files is not None:
            return json.load(open(files[0]))
        raise PreventUpdate

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       Output('session_alert_config', 'data'),
                       State('read-filepath', 'value'),
                       Input('add-file-by-path', 'n_clicks'),
                       State('session_config', 'data'),
                       State('session_alert_config', 'data'),
                       State('local-read-type', 'value'),
                       prevent_initial_call=True)
    def get_session_uploads_from_local_path(path, clicks, cur_session, error_config, import_type):
        if path and clicks > 0:
            session_config = cur_session if cur_session is not None and \
                            len(cur_session['uploads']) > 0 else {'uploads': []}
            error_config = {"error": None} if error_config is None else error_config
            if import_type == "filepath":
                if os.path.isfile(path):
                    session_config['uploads'].append(path)
                    error_config["error"] = None
                    return session_config, dash.no_update
                else:
                    error_config["error"] = ALERT.warnings["invalid_filepath"]
                    return dash.no_update, error_config
            elif import_type == "directory":
                if os.path.isdir(path):
                    # valid_files = []
                    extensions = ["*.tiff", "*.mcd", "*.tif", "*.txt", "*.h5"]
                    for extension in extensions:
                        session_config['uploads'].extend(Path(path).glob(extension))
                    session_config['uploads'] = [str(elem) for elem in session_config['uploads']]
                    return session_config, dash.no_update
                else:
                    error_config["error"] = ALERT.warnings["invalid_directory"]
                    return dash.no_update, error_config
            raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('session_config', 'data', allow_duplicate=True),
                       Input('local-dialog-file', 'n_clicks'),
                       State('session_config', 'data'),
                       prevent_initial_call=True)
    def read_from_local_dialog_box(nclicks, cur_session):
        if nclicks > 0:
            import wx
            app = wx.App(None)
            dialog = wx.FileDialog(None, 'Open', str(Path.home()),
                                   style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST,
                                   wildcard="*.tiff;*.tif;*.mcd;*.txt;*.h5|*.tiff;*.tif;*.mcd;*.txt;*.h5")
            if dialog.ShowModal() == wx.ID_OK:
                filenames = dialog.GetPaths()
                if filenames is not None and len(filenames) > 0 and isinstance(filenames, list):
                    session_config = cur_session if cur_session is not None and \
                                                    len(cur_session['uploads']) > 0 else {'uploads': []}
                    for filename in filenames:
                        session_config["uploads"].append(filename)
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
        session_config = cur_session if cur_session is not None and \
                                        len(cur_session['uploads']) > 0 else {'uploads': []}
        if upload_list is not None and len(upload_list) > 0:
            for new_upload in upload_list:
                if new_upload not in session_config["uploads"]:
                    session_config["uploads"].append(new_upload)
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
            error_config = {"error": None} if error_config is None else error_config
            files = natsorted(session_dict['uploads']) if natsort else session_dict['uploads']
            message, unique_suffixes = file_import_message(files)
            if len(unique_suffixes) > 1:
                error_config["error"] = ALERT.warnings["multiple_filetypes"] + message
            else:
                error_config["error"] = message
            fileparser = FileParser(files, array_store_type=app_config['array_store_type'], delimiter=delimiter)
            session_dict['unique_images'] = fileparser.unique_image_names
            columns = [{'id': p, 'name': p, 'editable': False} for p in fileparser.dataset_information_frame.keys()]
            data = pd.DataFrame(fileparser.dataset_information_frame).to_dict(orient='records')
            blend_return = fileparser.blend_config if current_blend is None or len(current_blend) == 0 else dash.no_update
            return SessionServerside(fileparser.image_dict, key="upload_dict",
                use_unique_key=app_config['serverside_overwrite']), session_dict, blend_return, columns, data, error_config
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
    # @cache.memoize())
    def populate_dataset_options(uploaded, cur_data_selection, cur_layers_selected):
        if uploaded is not None:
            datasets, selection_return, channels_return = [], None, None
            for roi in uploaded.keys():
                if "metadata" not in roi:
                    datasets.append(roi)
            if cur_data_selection is not None:
                selection_return = dash.no_update if cur_data_selection in datasets else None
                if cur_layers_selected is not None and len(cur_layers_selected) > 0:
                    channels_return = cur_layers_selected
            height_update = adjust_option_height_from_list_length(datasets)
            return datasets, selection_return, channels_return, height_update
            # TODO: decide if want to use an animation to draw attention to the data selection input
                # "animate__animated animate__jello animate__slower"
        raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options', allow_duplicate=True),
                       Output('data-collection', 'value', allow_duplicate=True),
                       Output('image_layers', 'options', allow_duplicate=True),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Input('remove-collection', 'n_clicks'),
                       State('data-collection', 'value'),
                       State('data-collection', 'options'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def remove_dataset_from_collection(remove_clicks, cur_data_selection, cur_options):
        """
        Use the trash icon to remove a dataset collection from the possible selections
        Causes a reset of the canvas, channel selection, and channel modification menus
        """
        return delete_dataset_option_from_list_interactively(remove_clicks, cur_data_selection, cur_options)

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Input('uploaded_dict_template', 'data'),
                       State('annotation_canvas', 'figure'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def reset_canvas_on_new_upload(uploaded, cur_fig):
        if None not in (uploaded, cur_fig) and 'data' in cur_fig:
            fig = go.Figure()
            fig['layout']['uirevision'] = True
            return fig
        raise PreventUpdate


    @dash_app.callback(Output('image_layers', 'options'),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('uploaded_dict', 'data', allow_duplicate=True),
                       Output('canvas-div-holder', 'children'),
                       Output('current-roi-ha', 'children'),
                       Output('cur_roi_dimensions', 'data'),
                       State('uploaded_dict_template', 'data'),
                       Input('data-collection', 'value'),
                       Input('alias-dict', 'data'),
                       State('image_layers', 'value'),
                       State('session_config', 'data'),
                       Input('sort-channels-alpha', 'value'),
                       State('enable-canvas-scroll-zoom', 'value'),
                       State('cur_roi_dimensions', 'data'),
                       Input('data-selection-refresh', 'n_clicks'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    def create_dropdown_options(image_dict, data_selection, names, currently_selected_channels, session_config,
                                sort_channels, enable_zoom, cur_dimensions, dataset_refresh, delimiter):
        # TODO: figure out if the canvas freezing occurs here when exceptions are thrown for canvas refreshing
        """
        Update the image layers and dropdown options when a new ROI is selected.
        Additionally, check the dimension of the incoming ROI, and wrap the annotation canvas in a load screen
        if the dimensions are above a specific pixel height and width for either axis
        """
        # set the default canvas to return without a load screen
        if image_dict and data_selection and names:
            exp, slide, roi_name = split_string_at_pattern(data_selection, pattern=delimiter)
            roi_name = str(roi_name) + f" ({str(exp)})" if "acq" in str(roi_name) else str(roi_name)
            if ' sort (A-z)' in sort_channels:
                channels_return = dict(sorted(names.items(), key=lambda x: x[1].lower()))
            else:
                channels_return = names
            if ctx.triggered_id not in ["sort-channels-alpha", "alias-dict"]:
                try:
                    image_dict = populate_image_dict_from_lazy_load(image_dict.copy(), dataset_selection=data_selection,
                    session_config=session_config, array_store_type=app_config['array_store_type'], delimiter=delimiter)
                    # check if the first image has dimensions greater than 3000. if yes, wrap the canvas in a loader
                    if data_selection in image_dict.keys() and all([image_dict[data_selection][elem] is not None for \
                        elem in image_dict[data_selection].keys()]):
                        # get the first image in the ROI and check the dimensions
                        first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
                        dim_return = (first_image.shape[0], first_image.shape[1])
                        # if the new dimensions match, do not update the canvas child to preserve the ui revision state
                        if cur_dimensions is not None and (first_image.shape[0] == cur_dimensions[0]) and \
                                (first_image.shape[1] == cur_dimensions[1]):
                           canvas_return = dash.no_update
                        else:
                            canvas_return = [wrap_canvas_in_loading_screen_for_large_images(first_image,
                            enable_zoom=enable_zoom, wrap=app_config['use_loading'], filename=data_selection,
                                        delimiter=delimiter)]
                    else:
                        canvas_return = [wrap_canvas_in_loading_screen_for_large_images(None, enable_zoom=enable_zoom,
                                        wrap=app_config['use_loading'], filename=data_selection, delimiter=delimiter)]
                # If there is an error on the dataset update, by default return a new fresh canvas
                except (IndexError, AssertionError, KeyError):
                    canvas_return = [wrap_canvas_in_loading_screen_for_large_images(None,
                                    enable_zoom=enable_zoom, wrap=app_config['use_loading'],
                                    filename=data_selection, delimiter=delimiter)]
                try:
                    # if all of the currently selected channels are in the new ROI, keep them. otherwise, reset
                    if currently_selected_channels is not None and len(currently_selected_channels) > 0 and \
                    all([elem in image_dict[data_selection].keys() for elem in currently_selected_channels]):
                        channels_selected = list(currently_selected_channels)
                    else:
                        channels_selected = []
                    return [{'label': names[i], 'value': i} for i in channels_return.keys() if len(i) > 0 and \
                        i not in ['', ' ', None]], channels_selected, \
                        SessionServerside(image_dict, key="upload_dict", use_unique_key=app_config['serverside_overwrite']), \
                        canvas_return, f"Current ROI: {roi_name}", dim_return
                except AssertionError:
                    return [], [], SessionServerside(image_dict, key="upload_dict", use_unique_key=
                    app_config['serverside_overwrite']), canvas_return, f"Current ROI: {roi_name}", dim_return
            elif ctx.triggered_id in ["sort-channels-alpha", "alias-dict"] and names is not None:
                return [{'label': names[i], 'value': i} for i in channels_return.keys() if len(i) > 0 and \
                        i not in ['', ' ', None]], dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
                    dash.no_update
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
        # TODO: decide if should have just the ROI name or the entire dataset identifier (current)
        if roi_change and current_canvas_config: return update_canvas_filename(current_canvas_config, roi_change, delim)
        raise PreventUpdate

    @dash_app.callback(Input('image_layers', 'options'),
                       Input('alias-dict', 'data'),
                       State('channel-quantification-list', 'value'),
                       Output('channel-quantification-list', 'options'),
                       Output('channel-quantification-list', 'value'),
                       prevent_initial_call=True)
    def create_channel_options_for_quantification(channel_options, aliases, cur_selection):
        """
        Create the dropdown options for the channels for quantification
        If channels are already selected, keep them and just update the labels
        """
        channel_list_options = [{'label': value, 'value': key} for key, value in aliases.items()]
        channel_list_selected = list(aliases.keys()) if not cur_selection else cur_selection
        return channel_list_options, channel_list_selected

    @dash_app.callback(Output('images_in_blend', 'options'),
                       Output('images_in_blend', 'value'),
                       Input('image_layers', 'value'),
                       Input('alias-dict', 'data'),
                       State('images_in_blend', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def create_dropdown_blend(chosen_for_blend, names, cur_channel_mod):
        """
        Create the dropdown menu for the channel modification menu on layer changes
        Auto-fill the value with the latest channel if it doesn't match the current value in the modification
        """
        if chosen_for_blend is not None and len(chosen_for_blend) > 0:
            try:
                if not all([elem in names.keys() for elem in chosen_for_blend]): raise AssertionError
                channel_auto_fill = dash.no_update
                if chosen_for_blend[-1] != cur_channel_mod:
                    channel_auto_fill = chosen_for_blend[-1]
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
                       State('image_layers', 'value'),
                       State('canvas-layers', 'data'),
                       State('blending_colours', 'data'),
                       State('session_alert_config', 'data'),
                       State('db-saved-configs', 'data'),
                       State("imc-metadata-editable", "data"),
                       State('dataset-delimiter', 'value'),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('bool-apply-global-filter', 'value', allow_duplicate=True),
                       Output('global-filter-type', 'value', allow_duplicate=True),
                       Output("global-kernel-val-filter", 'value', allow_duplicate=True),
                       Output("global-sigma-val-filter", 'value', allow_duplicate=True),
                       Output("imc-metadata-editable", "data", allow_duplicate=True),
                       Output('db-config-options', 'value', allow_duplicate=True),
                       Output('db-config-name', 'value', allow_duplicate=True),
                       Output('cluster-colour-assignments-dict', 'data', allow_duplicate=True),
                       Output('gating-dict', 'data', allow_duplicate=True),
                       prevent_initial_call=True)
    def update_parameters_from_config_json_or_db(uploaded_w_data, new_blend_dict, db_config_selection, data_selection,
            add_to_layer, all_layers, current_blend_dict, error_config, db_config_list, cur_metadata, delimiter):
        """
        Update the blend layer dictionary and currently selected channels from a JSON upload
        Only applies to the channels that have already been selected: if channels are not in the current blend,
        they will be modified on future selection
        Requires that the channel modification menu be empty to make sure that parameters are updated properly
        """
        error_config = {"error": None} if error_config is None else error_config
        if ctx.triggered_id == "db-config-options" and db_config_selection is not None:
            # TODO: decide if the alias key needs to be removed from the blend dict imported from mongoDB
            new_blend_dict = match_db_config_to_request_str(db_config_list, db_config_selection)
        metadata_return = extract_alias_labels_from_db_document(new_blend_dict, cur_metadata)
        metadata_return = metadata_return if len(metadata_return) > 0 else dash.no_update
        if None not in (uploaded_w_data, new_blend_dict, data_selection):
            # conditions where the blend dictionary is updated
            # reformat the blend dict to remove the metadata key if reported with h5py so it will match
            current_blend_dict = {key: value for key, value in current_blend_dict.items() if 'metadata' not in key}
            panels_equal = current_blend_dict is not None and len(current_blend_dict) == len(new_blend_dict['channels'])
            match_all = current_blend_dict is None and all([len(uploaded_w_data[roi]) == \
                        len(new_blend_dict['channels']) for roi in uploaded_w_data.keys() if delimiter in roi])
            if panels_equal or match_all:
                current_blend_dict = new_blend_dict['channels'].copy()
                if all_layers is None or data_selection not in all_layers.keys():
                    all_layers = {data_selection: {}}
                for elem in add_to_layer:
                    current_blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                        current_blend_dict, elem, uploaded_w_data, data_selection)
                    array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem], current_blend_dict[elem])
                    all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                    current_blend_dict[elem]['color'])).astype(np.uint8)
                error_config["error"] = ALERT.warnings["json_update_success"]
                channel_list_return = dash.no_update
                if 'config' in new_blend_dict and 'blend' in new_blend_dict['config'] and all([elem in \
                        current_blend_dict.keys() for elem in new_blend_dict['config']['blend']]):
                    channel_list_return = new_blend_dict['config']['blend']
                global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma = \
                parse_global_filter_values_from_json(new_blend_dict['config'])
                clust_return = {data_selection: new_blend_dict['cluster']} if \
                    'cluster' in new_blend_dict.keys() and new_blend_dict['cluster'] else dash.no_update
                gate_return = new_blend_dict['gating'] if 'gating' in new_blend_dict.keys() else dash.no_update
                return SessionServerside(all_layers, key="layer_dict", use_unique_key=app_config['serverside_overwrite']), \
                    current_blend_dict, error_config, channel_list_return, global_apply_filter, global_filter_type, \
                    global_filter_val, global_filter_sigma, metadata_return, dash.no_update, dash.no_update, clust_return, gate_return
            # IMP: if the update does not occur, clear the database selection and auto filled config name
            else:
                error_config["error"] = ALERT.warnings["json_update_error"]
                return dash.no_update, dash.no_update, error_config, dash.no_update, dash.no_update, \
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update
        elif data_selection is None:
            error_config["error"] = ALERT.warnings["json_requires_roi"]
            return dash.no_update, dash.no_update, error_config, dash.no_update, dash.no_update, \
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update
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
        use_preset_condition = None not in (preset_selection, preset_dict)
        if add_to_layer is not None and current_blend_dict is not None:
            channel_modify = dash.no_update
            if param_dict is None or len(param_dict) < 1:
                param_dict = {"current_roi": data_selection}
            if data_selection is not None:
                if current_blend_dict is not None and "current_roi" in param_dict.keys() and \
                        data_selection != param_dict["current_roi"]:
                    # current_blend_dict = copy_values_within_nested_dict(current_blend_dict, param_dict["current_roi"],
                    #                                                     data_selection)
                    param_dict["current_roi"] = data_selection
                    if cur_image_in_mod_menu is not None and cur_image_in_mod_menu in current_blend_dict.keys():
                        channel_modify = cur_image_in_mod_menu
                else:
                    param_dict["current_roi"] = data_selection
            if all_layers is None or data_selection not in all_layers.keys():
                all_layers = {data_selection: {}}
            for elem in add_to_layer:
                # if the selected channel doesn't have a config yet, create one either from scratch or a preset
                if elem not in current_blend_dict.keys():
                    current_blend_dict[elem] = {'color': None, 'x_lower_bound': 0, 'x_upper_bound':
                        get_default_channel_upper_bound_by_percentile(uploaded_w_data[data_selection][elem]),
                            'filter_type': None, 'filter_val': None, 'filter_sigma': None}
                    # TODO: default colour is white, but can set auto selection here for starting colours
                    current_blend_dict[elem]['color'] = '#FFFFFF'
                    if autofill_channel_colours:
                        current_blend_dict = select_random_colour_for_channel(current_blend_dict, elem, DEFAULT_COLOURS)
                    if use_preset_condition:
                        current_blend_dict[elem] = apply_preset_to_blend_dict(
                            current_blend_dict[elem], preset_dict[preset_selection])
                # if the selected channel is in the current blend, check if a preset is used to override
                elif elem in current_blend_dict.keys() and use_preset_condition:
                    # do not override the colour of the current channel
                    current_blend_dict[elem] = apply_preset_to_blend_dict(
                        current_blend_dict[elem], preset_dict[preset_selection])
                else:
                    # TODO: default colour is white, but can set auto selection here for starting colours
                    if autofill_channel_colours:
                        current_blend_dict = select_random_colour_for_channel(current_blend_dict, elem, DEFAULT_COLOURS)
                    # create a nested dict with the image and all of the filters being used for it
                    # if the same blend parameters have been transferred from another ROI, apply them
                    # set a default upper bound for the channel if the value is None
                    # if current_blend_dict[elem]['x_upper_bound'] is None:
                    #     current_blend_dict[elem]['x_upper_bound'] = \
                    #     get_default_channel_upper_bound_by_percentile(uploaded_w_data[data_selection][elem])
                    # if current_blend_dict[elem]['x_lower_bound'] is None:
                    #     current_blend_dict[elem]['x_lower_bound'] = 0
                    current_blend_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                        current_blend_dict, elem, uploaded_w_data, data_selection)
                    # TODO: evaluate whether there should be a conditional here if the elem is already
                    #  present in the layers dictionary to save time
                    # affects if a channel is added and dropped
                    if (data_selection in all_layers.keys() and elem not in all_layers[data_selection].keys()) or \
                            autofill_channel_colours:
                        array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem],
                                                         current_blend_dict[elem])
                        all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                            current_blend_dict[elem]['color'])).astype(np.uint8)

            return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                use_unique_key=app_config['serverside_overwrite']), param_dict, channel_modify
        raise PreventUpdate

    @dash_app.callback(Output("annotation-color-picker", 'value', allow_duplicate=True),
                       Output('swatch-color-picker', 'value'),
                       Input('swatch-color-picker', 'value'),
                       prevent_initial_call=True)
    def update_colour_picker_from_swatch(swatch):
        if swatch is not None:
            # IMP: need to reset the value of the swatch to None after transferring the colour
            return dict(hex=swatch), None
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
    # @cache.memoize())
    def update_blend_dict_on_color_selection(colour, layer, uploaded_w_data,
                                    current_blend_dict, data_selection, add_to_layer,
                                    all_layers, filter_chosen, filter_name, filter_value, filter_sigma,
                                    blend_options):
        """
        Update the blend dictionary and layer dictionary when a modification channel changes its colour
        """
        if layer is not None and current_blend_dict is not None and data_selection is not None and \
                current_blend_dict is not None:

            array = uploaded_w_data[data_selection][layer]
            if current_blend_dict[layer]['color'] != colour['hex']:
                blend_options = [elem['value'] for elem in blend_options]
                if all([elem in add_to_layer for elem in blend_options]):

                    # if upper and lower bounds have been set before for this layer, use them before recolouring

                    if current_blend_dict[layer]['x_lower_bound'] is not None and \
                        current_blend_dict[layer]['x_upper_bound'] is not None:
                        array = filter_by_upper_and_lower_bound(array, float(current_blend_dict[layer]['x_lower_bound']),
                                float(current_blend_dict[layer]['x_upper_bound']))

                    if len(filter_chosen) > 0 and filter_name is not None:
                        if filter_name == "median" and int(filter_value) >= 1:
                            try:
                                array = median_filter(array, int(filter_value))
                            except ValueError:
                                pass
                        else:
                            # array = gaussian_filter(array, int(filter_value))
                            if int(filter_value) % 2 != 0 and int(filter_value) >= 1:
                                array = cv2.GaussianBlur(array, (int(filter_value),
                                                                 int(filter_value)), float(filter_sigma))
                    current_blend_dict[layer]['color'] = colour['hex']
                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                        colour['hex'])).astype(np.uint8)
                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                                                use_unique_key=app_config['serverside_overwrite'])
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
    # @cache.memoize())
    def set_blend_params_on_pixel_range_adjustment(layer, uploaded_w_data,
                                    current_blend_dict, data_selection,
                                    all_layers, slider_values):

        if None not in (slider_values, layer, data_selection, uploaded_w_data) and \
                all([elem is not None for elem in slider_values]):
            # do not update if the range values in the slider match the current blend params:
            try:
                slider_values = [float(elem) for elem in slider_values]
                lower_bound, upper_bound = min(slider_values), max(slider_values)

                if float(current_blend_dict[layer]['x_lower_bound']) == float(lower_bound) and \
                        float(current_blend_dict[layer]['x_upper_bound']) == float(upper_bound):
                    raise PreventUpdate
                else:
                    current_blend_dict[layer]['x_lower_bound'] = float(lower_bound)
                    current_blend_dict[layer]['x_upper_bound'] = float(upper_bound)

                    array = apply_preset_to_array(uploaded_w_data[data_selection][layer], current_blend_dict[layer])

                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                    current_blend_dict[layer]['color']))

                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                                                                 use_unique_key=app_config['serverside_overwrite'])
            except TypeError:
                raise PreventUpdate
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
    # @cache.memoize())
    def set_blend_params_on_preset_selection(layer, uploaded_w_data,
                                    current_blend_dict, data_selection,
                                    all_layers, preset_selection, preset_dict):
        """
        Set the blend param dictionary and canvas layer dictionary when a preset is applied to the current ROI.
        """
        preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val', 'filter_sigma']
        if None not in (preset_selection, preset_dict, data_selection, current_blend_dict, layer):

            array = uploaded_w_data[data_selection][layer]

            for preset_val in preset_keys:
                current_blend_dict[layer][preset_val] = preset_dict[preset_selection][preset_val]

            array = apply_preset_to_array(array, preset_dict[preset_selection])
            all_layers[data_selection][layer] = np.array(recolour_greyscale(array, current_blend_dict[layer]['color']))
            return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                                    use_unique_key=app_config['serverside_overwrite'])

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
    # @cache.memoize())
    def set_blend_options_for_layer_with_bool_filter(layer, uploaded, current_blend_dict, data_selection,
                                                     all_layers, filter_chosen, filter_name, filter_value, filter_sigma,
                                                     cur_layers, blend_options, session_vars):

        only_options_changed = False
        if None not in (ctx.triggered, session_vars):
            # do not update if the trigger is the channel options and the current selection hasn't changed
            only_options_changed = ctx.triggered_id == "images_in_blend" and \
                                   ctx.triggered[0]['value'] == session_vars["cur_channel"]

        if None not in (layer, current_blend_dict, data_selection, filter_value, filter_name, all_layers,
                        filter_sigma) and not only_options_changed:

            try:
                array = uploaded[data_selection][layer]
            except KeyError:
                array = None

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
                        if filter_name == "median" and int(filter_value) >= 1:
                            try:
                                array = median_filter(array, int(filter_value))
                            except ValueError:
                                pass
                        else:
                            # array = gaussian_filter(array, int(filter_value))
                            if int(filter_value) % 2 != 0:
                                array = cv2.GaussianBlur(array, (int(filter_value),
                                            int(filter_value)), float(filter_sigma))

                        current_blend_dict[layer]['filter_type'] = filter_name
                        current_blend_dict[layer]['filter_val'] = filter_value
                        current_blend_dict[layer]['filter_sigma'] = filter_sigma

                    else:
                        current_blend_dict[layer]['filter_type'] = None
                        current_blend_dict[layer]['filter_val'] = None
                        current_blend_dict[layer]['filter_sigma'] = None

                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                        current_blend_dict[layer]['color'])).astype(np.uint8)

                    return current_blend_dict, SessionServerside(all_layers, key="layer_dict",
                                                use_unique_key=app_config['serverside_overwrite'])
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
        if data_selection: return None
        raise PreventUpdate

    @dash_app.callback(Output('blending_colours', 'data', allow_duplicate=True),
                       Input('preset-options', 'value'),
                       Input('image_presets', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def set_blend_options_from_preset(preset_selection, preset_dict, current_blend_dict, data_selection):
        if None not in (preset_selection, preset_dict, current_blend_dict, data_selection):

            for key, value in current_blend_dict.items():
                current_blend_dict[key] = apply_preset_to_blend_dict(value, preset_dict[preset_selection])
            return current_blend_dict
        raise PreventUpdate

    # @dash_app.callback(Output('image_layers', 'value', allow_duplicate=True),
    #                    Input('data-collection', 'value'),
    #                    State('image_layers', 'value'),
    #                    prevent_initial_call=True)
    # # @cache.memoize())
    # def reset_image_layers_selected(current_layers, new_selection):
    #     if new_selection is not None and current_layers is not None:
    #         if len(current_layers) > 0:
    #             return None
    #     else:
    #         raise PreventUpdate

    # @dash_app.callback(Output('images_in_blend', 'value', allow_duplicate=True),
    #                    Output('images_in_blend', 'options', allow_duplicate=True),
    #                    Input('data-collection', 'value'),
    #                    # State('image_layers', 'value'),
    #                    prevent_initial_call=True)
    # # @cache.memoize())
    # def reset_blend_layers_selected(new_selection):
    #     """
    #     Reset the channels that are available in the modification dropdown manu when a new ROI/dataset is selected.
    #     """
    #     if new_selection is not None:
    #         return None, []
    #     else:
    #         raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Output('mask-options', 'value', allow_duplicate=True),
                       Input('data-collection', 'value'),
                       State('data-collection', 'options'),
                       State('mask-options', 'options'),
                       # State('image_layers', 'value'),
                       State('annotation_canvas', 'figure'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def clear_canvas_and_set_mask_on_new_dataset(new_selection, dataset_options, mask_options, cur_canvas, delimiter):
        """
        Reset the canvas to blank on an ROI change
        Will attempt to set the new mask based on the ROI name and the list of mask options
        """
        #TODO: new update here does not reset the canvas to blank between ROI selections for smoother transition
        if new_selection is not None:
            canvas_return = dash.no_update
            if 'shapes' in cur_canvas['layout'] and len(cur_canvas['layout']['shapes']) > 0:
                other_shapes = [shape for shape in cur_canvas['layout']['shapes'] if \
                            shape is not None and 'type' in shape and (shape['type'] in ['path', 'rect', 'circle'] or \
                            any(elem in ['rect', 'path', 'circle'] for elem in shape.keys()))]
                if len(other_shapes) > 0:
                    for shape in cur_canvas['layout']['shapes']:
                        if 'label' in shape and 'texttemplate' in shape['label']:
                            shape['label'].pop('texttemplate')
                    canvas_return = cur_canvas
            return canvas_return, match_mask_name_with_roi(new_selection, mask_options, dataset_options, delimiter)
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

    @dash_app.callback(Output('annotation_canvas', 'figure'),
                       # Output('annotation_canvas', 'relayoutData'),
                       Output('download-canvas-image-tiff', 'data'),
                       # Output('data-collection', 'value', allow_duplicate=True),
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
                       State('imported-cluster-frame', 'data'),
                       Input('cluster-annotation-type', 'value'),
                       Input('btn-download-canvas-tiff', 'n_clicks'),
                       State('custom-scale-val', 'value'),
                       State('cluster-annotations-legend', 'value'),
                       Input('apply-gating', 'value'),
                       Input('gating-cell-list', 'data'),
                       State('dataset-delimiter', 'value'),
                       State('scalebar-color', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def render_canvas_from_layer_mask_hover_change(canvas_layers, currently_selected,
                                                data_selection, blend_colour_dict, aliases,
                                                cur_graph, cur_graph_layout, raw_data_dict,
                                                show_each_channel_intensity,
                                                param_dict, mask_config, mask_toggle,
                                                mask_selection, toggle_legend, toggle_scalebar, mask_blending_level,
                                                add_mask_boundary, channel_order, legend_size, add_cell_id_hover,
                                                pixel_ratio, invert_annot, overlay_grid, legend_orientation,
                                                global_apply_filter, global_filter_type, global_filter_val,
                                                global_filter_sigma, apply_cluster_on_mask, cluster_assignments_dict,
                                                cluster_frame, cluster_type, download_canvas_tiff, custom_scale_val,
                                                cluster_assignments_in_legend, apply_gating, gating_cell_id_list,
                                                delimiter, scale_color):

        """
        Update the canvas from either an underlying change to the source image, or a change to the hover template
        Examples of triggers include:
        - changes to the additive blend (channel modifications & global filters)
        - toggling of a mask and its related features (blending, clustering)
        - if the hover template is updated (it is faster to recreate the figure rather than trying to remove the
        hover template)
        """
        # TODO: decide if an error should prompt a recursive try on the canvas by re-sending the data selection
        # do not update if the trigger is a global filter and the filter is not applied
        global_not_enabled = ctx.triggered_id in ["global-filter-type", "global-kernel-val-filter",
                                                  "global-sigma-val-filter"] and not global_apply_filter
        channel_order_same = ctx.triggered_id in ["channel-order"] and channel_order == currently_selected
        if canvas_layers is not None and currently_selected is not None and blend_colour_dict is not None and \
                data_selection is not None and currently_selected and len(channel_order) > 0 and not global_not_enabled \
                and not channel_order_same and data_selection in canvas_layers and canvas_layers[data_selection]:
                # data_selection in canvas_layers:
            cur_graph = strip_invalid_shapes_from_graph_layout(cur_graph)
            pixel_ratio = pixel_ratio if pixel_ratio is not None else 1
            legend_text = generate_canvas_legend_text(blend_colour_dict, channel_order, aliases, legend_orientation,
                        cluster_assignments_in_legend, cluster_assignments_dict, data_selection)
            try:
                canvas = CanvasImage(canvas_layers, data_selection, currently_selected,
                 mask_config, mask_selection, mask_blending_level,
                 overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                 legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                 show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter,
                global_filter_type, global_filter_val, global_filter_sigma,
                apply_cluster_on_mask, cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val,
                                     apply_gating, gating_cell_id_list, scale_color)
                fig = canvas.generate_canvas()
                if cluster_type == 'mask' or not apply_cluster_on_mask:
                    fig = CanvasLayout(fig)
                    fig = fig.remove_cluster_annotation_shapes()
                elif apply_cluster_on_mask:
                    fig = CanvasLayout(fig)
                    fig = fig.add_cluster_annotations_as_circles(mask_config[mask_selection]["raw"], pd.DataFrame(
                cluster_frame[data_selection]), cluster_assignments_dict, data_selection, 2, apply_gating, gating_cell_id_list)
                # set if the image is to be downloaded or not
                dest_path = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
                canvas_tiff = dash.no_update
                if ctx.triggered_id == "btn-download-canvas-tiff":
                    fig = dash.no_update
                    canvas_tiff = dcc.send_file(output_current_canvas_as_tiff(canvas_image=canvas.get_image(),
                        dest_dir=dest_path, use_roi_name=True, roi_name=data_selection, delimiter=delimiter))
                return fig, canvas_tiff
            except (ValueError, AttributeError, KeyError, IndexError):
                raise PreventUpdate
        #TODO: this step can be used to keep the current ui revision if a new ROI is selected with the same dimensions
        # elif currently_selected is not None and 'shapes' not in cur_graph_layout:
        #     fig = cur_graph if cur_graph is not None else go.Figure()
        #     if 'data' in fig:
        #         fig['data'] = []
        #         if 'shapes' in fig['layout']:
        #             fig['layout']['shapes'] = []
        #         if 'annotations' in fig['layout']:
        #             fig['layout']['annotations'] = [annotation for annotation in \
        #                                         fig['layout']['annotations'] if annotation['y'] == 0.06]
        #     fig = go.Figure(fig)
        #     fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
        #                       xaxis=XAxis(showticklabels=False),
        #                       yaxis=YAxis(showticklabels=False),
        #                       margin=dict(
        #                           l=0,
        #                           r=0,
        #                           b=0,
        #                           t=0,
        #                           pad=0
        #                       ), dragmode="zoom")
        #     return fig, None, {'autosize': True}

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
    # @cache.memoize())
    def render_canvas_from_coord_or_zoom_change(cur_graph, cur_graph_layout, x_request, y_request, current_window,
                             nclicks_coord, data_selection, image_dict, custom_scale_val, pixel_ratio,
                            toggle_scalebar, legend_size, invert_annot, scale_col):

        """
        Update the annotation canvas when the zoom or custom coordinates are requested.
        """
        # bad_update = cur_graph_layout in [{"autosize": True}]

        # update the scale bar with and without zooming
        if None not in (cur_graph, cur_graph_layout, data_selection):
            cur_graph = strip_invalid_shapes_from_graph_layout(cur_graph)
            pixel_ratio = pixel_ratio if pixel_ratio is not None else 1
            if ctx.triggered_id not in ["activate-coord"]:
                try:
                    image_shape = get_first_image_from_roi_dictionary(image_dict[data_selection]).shape
                    proportion = float(custom_scale_val / image_shape[1]) if custom_scale_val is not None else 0.1
                    # if ctx.triggered_id == 'annotation_canvas':
                    cur_graph = CanvasLayout(cur_graph).update_scalebar_zoom_value(cur_graph_layout,
                                                    pixel_ratio, proportion, scale_col)
                    # if ctx.triggered_id == "custom-scale-val":
                    pixel_ratio = pixel_ratio if pixel_ratio is not None else 1
                    x_axis_placement = set_x_axis_placement_of_scalebar(image_shape[1], invert_annot)
                    cur_graph = CanvasLayout(cur_graph).toggle_scalebar(toggle_scalebar, x_axis_placement, invert_annot,
                                            pixel_ratio, image_shape, legend_size, proportion, scale_col)
                    # elif ctx.triggered_id == 'pixel-size-ratio':
                    #     cur_graph = CanvasLayout(cur_graph).use_custom_scalebar_value(custom_scale_val, pixel_ratio, proportion)
                    return cur_graph, cur_graph_layout
                except (ValueError, KeyError, AssertionError):
                    raise PreventUpdate
            if ctx.triggered_id == "activate-coord":
                if None not in (x_request, y_request, current_window) and \
                        nclicks_coord is not None and nclicks_coord > 0:
                    try:
                        fig, new_layout = CanvasLayout(cur_graph).update_coordinate_window(current_window, x_request, y_request)
                        return fig, new_layout
                    except (AssertionError, TypeError):
                        raise PreventUpdate
                raise PreventUpdate
            raise PreventUpdate
        raise PreventUpdate

    # @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
    #                    State('annotation_canvas', 'figure'),
    #                    Input('annotation_canvas', 'relayoutData'),
    #                    prevent_initial_call=True)
    # # @cache.memoize())
    # def fix_x_coords(canvas, layout):
    #     zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
    #     if all([elem in layout for elem in zoom_keys]) and canvas is not None and 'layout' in canvas:
    #         if canvas['layout']['xaxis']['range'][0] != layout['xaxis.range[0]'] and \
    #                 canvas['layout']['xaxis']['range'][1] != layout['xaxis.range[1]']:
    #             print("fixing")
    #             fig = go.Figure(canvas)
    #             fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
    #                               yaxis=YAxis(showticklabels=False,
    #                                           range=[layout['yaxis.range[1]'], layout['yaxis.range[0]']]),
    #                               xaxis=XAxis(showticklabels=False,
    #                                           range=[layout['xaxis.range[0]'], layout['xaxis.range[1]']]))
    #             return fig
    #         else:
    #             raise PreventUpdate
    #     else:
    #         raise PreventUpdate


    # @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
    #                    State('annotation_canvas', 'figure'),
    #                    State('annotation_canvas', 'relayoutData'),
    #                    Input('custom-scale-val', 'value'),
    #                    Input('pixel-size-ratio', 'value'),
    #                    prevent_initial_call=True)
    # # @cache.memoize())
    # def render_canvas_from_scalebar_change(cur_graph, cur_graph_layout, custom_scale_val, pixel_ratio):
    #     # do not update the canvas if the pixel ratio is changed to None
    #     pixel_ratio_none = ctx.triggered_id == 'pixel-size-ratio' and pixel_ratio is None
    #     if cur_graph is not None and cur_graph_layout not in [{'dragmode': 'pan'}] and not pixel_ratio_none:
    #         try:
    #             # TODO: change the scalebar length in conjunction with the value
    #             fig = CanvasLayout(cur_graph).use_custom_scalebar_value(custom_scale_val, pixel_ratio)
    #         except (KeyError, AssertionError):
    #             fig = dash.no_update
    #         return fig
    #     else:
    #         raise PreventUpdate

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
                       Input('scalebar-color', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_toggle_show_annotations(toggle_legend, toggle_scalebar,
                                                   cur_canvas, cur_layout, currently_selected,
                                                   data_selection, blend_colour_dict, aliases, image_dict,
                                                   channel_order, legend_size, pixel_ratio, invert_annot,
                                                   legend_orientation, custom_scale_val, cluster_assignments_in_legend,
                                                   cluster_assignments_dict, scalebar_col):
        """
        re-render the canvas if the user requests to remove the annotations (scalebar and legend) or
        updates the scalebar length with a custom value
        """
        if None not in (cur_layout, cur_canvas, data_selection, currently_selected, blend_colour_dict):
            pixel_ratio = pixel_ratio if pixel_ratio is not None else 1
            image_shape = get_first_image_from_roi_dictionary(image_dict[data_selection]).shape
            x_axis_placement = set_x_axis_placement_of_scalebar(image_shape[1], invert_annot)
            cur_canvas = CanvasLayout(cur_canvas).clear_improper_shapes()
            if ctx.triggered_id in ["toggle-canvas-legend", "legend_orientation", "cluster-annotations-legend", "channel-order"]:
                legend_text = generate_canvas_legend_text(blend_colour_dict, channel_order, aliases, legend_orientation,
                cluster_assignments_in_legend, cluster_assignments_dict, data_selection) if toggle_legend else ''
                canvas = CanvasLayout(cur_canvas).toggle_legend(toggle_legend, legend_text, x_axis_placement, legend_size)
                return CanvasLayout(canvas).clear_improper_shapes()
            elif ctx.triggered_id in ["toggle-canvas-scalebar", "scalebar-color"]:
                proportion = float(custom_scale_val / image_shape[1]) if custom_scale_val is not None else 0.1
                canvas = CanvasLayout(cur_canvas).toggle_scalebar(toggle_scalebar, x_axis_placement, invert_annot,
                        pixel_ratio, image_shape, legend_size, proportion, scalebar_col)
                return CanvasLayout(canvas).clear_improper_shapes()
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
            cur_canvas = strip_invalid_shapes_from_graph_layout(cur_canvas)
            return invert_annotations_figure(cur_canvas)
        raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       Input('legend-size-slider', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def render_canvas_from_change_annotation_size(cur_graph, legend_size):
        """
        Update the canvas when the size of the annotations is modified
        """
        if cur_graph is not None:
            try:
                return CanvasLayout(cur_graph).change_annotation_size(legend_size)
            except KeyError:
                raise PreventUpdate
        raise PreventUpdate

    @du.callback(Output('metadata_config', 'data'),
                 id='upload-metadata')
    # @cache.memoize())
    def upload_custom_metadata_panel(status: du.UploadStatus):
        """
        Upload a metadata panel separate from the auto-generated metadata panel. This must be parsed against the existing
        datasets to ensure that it matches the number of channels
        """
        metadata_config = {'uploads': []}
        uploader = DashUploaderFileReader(status)
        files = uploader.return_filenames()
        if files:
            for file in files:
                metadata_config['uploads'].append(file)
            return metadata_config
        raise PreventUpdate

    @dash_app.callback(
        Output("imc-metadata-editable", "columns", allow_duplicate=True),
        Output("imc-metadata-editable", "data", allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Input('metadata_config', 'data'),
        State('uploaded_dict_template', 'data'),
        State('session_alert_config', 'data'),
        State('imc-metadata-editable', 'data'),
        prevent_initial_call=True)
    # @cache.memoize())
    def populate_datatable_columns(metadata_config, uploaded, error_config, cur_metadata):
        if metadata_config is not None and len(metadata_config['uploads']) > 0:
            error_config = {"error": None} if error_config is None else error_config
            metadata_read = pd.read_csv(metadata_config['uploads'][0])
            metadata_validated = validate_incoming_metadata_table(metadata_read, uploaded)
            if metadata_validated is not None and 'ccramic Label' not in metadata_validated.keys():
                # TODO: decide if overwrite existing metadata, or just replace editable labels
                # make sure that the internal keys from channel names stay the same
                metadata_validated['Channel Name'] = pd.DataFrame(cur_metadata)['Channel Name']
                metadata_validated['ccramic Label'] = metadata_validated["Channel Label"]
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                        p in metadata_validated.keys()], \
                    pd.DataFrame(metadata_validated).to_dict(orient='records'), dash.no_update
            else:
                error_config["error"] = ALERT.warnings["custom_metadata_error"]
                return dash.no_update, dash.no_update, error_config
        raise PreventUpdate

    @dash_app.callback(
        Output("imc-metadata-editable", "columns"),
        Output("imc-metadata-editable", "data"),
        Input('uploaded_dict_template', 'data'),
        Input('image-metadata', 'data'))
    # @cache.memoize())
    def populate_metadata_table(uploaded, column_dict):
        if uploaded is not None and uploaded['metadata'] is not None:
            try:
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                    p in uploaded['metadata'].keys()], pd.DataFrame(uploaded['metadata']).to_dict(orient='records')
            except ValueError:
                raise PreventUpdate
        elif column_dict is not None:
            return column_dict["columns"], column_dict["data"]
        raise PreventUpdate

    @dash_app.callback(
        Input("imc-metadata-editable", "data"),
        Output('alias-dict', 'data'))
    # @cache.memoize())
    def create_channel_label_dict(metadata):
        if metadata is not None:
            return populate_alias_dict_from_editable_metadata(metadata)

    @dash_app.callback(
        Output("download-edited-table", "data"),
        Input("btn-download-metadata", "n_clicks"),
        Input("imc-metadata-editable", "data"))
    # @cache.memoize())
    def download_edited_metadata(n_clicks, datatable_contents):
        if n_clicks is not None and n_clicks > 0 and datatable_contents is not None and \
                ctx.triggered_id == "btn-download-metadata":
            return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "metadata.csv", index=False)
        raise PreventUpdate

    @dash_app.callback(Output('download-canvas-image-html', 'data'),
                       Input('btn-download-canvas-html', 'n_clicks'),
                       State('annotation_canvas', 'figure'),
                       State('uploaded_dict', 'data'),
                       State('blending_colours', 'data'),
                       State('annotation_canvas', 'style'),
                       State('data-collection', 'value'),
                       State('dataset-delimiter', 'value'))
    # @cache.memoize())
    def download_interactive_html_canvas(download_html, cur_graph, uploaded, blend_dict, canvas_style,
                                         dataset_selection, delimiter):
        if None not in (cur_graph, uploaded, blend_dict) and download_html > 0:
            try:
                download_dir = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
                create_download_dir(download_dir)
                html_path = dcc.send_file(output_current_canvas_as_html(cur_graph, canvas_style, download_dir,
                                        use_roi_name=True, roi_name=dataset_selection, delimiter=delimiter))
            except (ValueError, KeyError):
                html_path = dash.no_update
            return html_path
        raise PreventUpdate

    # @dash_app.callback(Output('download-canvas-image-tiff-annotations', 'data'),
    #                    Input('btn-download-canvas-tiff-annotations', 'n_clicks'),
    #                    State('annotation_canvas', 'figure'),
    #                    State('data-collection', 'value'),
    #                    State('dataset-delimiter', 'value'),
    #                    State('uploaded_dict', 'data'))
    # # @cache.memoize())
    # def download_tiff_w_annotations(download_tiff_annotations, cur_graph, data_selection, delimiter, uploaded):
    #     if None not in (cur_graph, uploaded, data_selection) and download_tiff_annotations > 0:
    #         dest_path = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
    #         first_image = get_first_image_from_roi_dictionary(uploaded[data_selection])
    #         fig = go.Figure(cur_graph)
    #         fig.update_layout(margin=dict(
    #                     l=0,
    #                     r=0,
    #                     b=0,
    #                     t=0,
    #                     pad=0
    #                 ))
    #         canvas = plotly_fig2array(fig, first_image)
    #         return dcc.send_file(output_current_canvas_as_tiff(canvas_image=canvas,
    #                 dest_dir=dest_path, use_roi_name=True,roi_name=data_selection, delimiter=delimiter))
    #     raise PreventUpdate

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
                       State('gating-dict', 'data'))
    # @cache.memoize())
    def download_session_config_json(download_json, blend_dict, blend_layers, global_apply_filter,
        global_filter_type, global_filter_val, global_filter_sigma, cluster_assignments, data_selection, aliases, gating_dict):
        if blend_dict is not None and download_json > 0:
            download_dir = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
            create_download_dir(download_dir)
            return dcc.send_file(write_blend_config_to_json(download_dir, blend_dict, blend_layers, global_apply_filter,
            global_filter_type, global_filter_val, global_filter_sigma, data_selection, cluster_assignments, aliases, gating_dict))
        raise PreventUpdate

    @dash_app.callback(Output('download-roi-h5py', 'data'),
                       # Output('download-canvas-interactive-html', 'href'),
                       Input('btn-download-roi-h5py', 'n_clicks'),
                       State('uploaded_dict', 'data'),
                       State('imc-metadata-editable', 'data'),
                       State('blending_colours', 'data'),
                       State('data-collection', 'value'),
                       State('annotation_canvas', 'relayoutData'),
                       State('graph-subset-download', 'value'))
    # @cache.memoize())
    def update_download_href_h5(download_h5py, uploaded, metadata_sheet, blend_dict, data_selection,
                                canvas_layout, graph_subset):
        """
        Create the download links for the current canvas and the session data.
        Only update if the download dialog is open to avoid continuous updating on canvas change
        """
        if None not in (uploaded, blend_dict) and download_h5py > 0:
            first_image = get_first_image_from_roi_dictionary(uploaded[data_selection])
            download_dir = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
            create_download_dir(download_dir)
            try:
                mask = None
                if 'shapes' in canvas_layout and ' use graph subset on download' in graph_subset:
                    mask = subset_mask_for_data_export(canvas_layout, first_image.shape)
                return dcc.send_file(write_session_data_to_h5py(download_dir, metadata_sheet, uploaded,
                                                               data_selection, blend_dict, mask))
            # if the dictionary hasn't updated to include all the experiments, then don't update download just yet
            except KeyError:
                raise PreventUpdate
        raise PreventUpdate


    @dash_app.callback(
        Output('annotation_canvas', 'style'),
        Input('annotation-canvas-size', 'value'),
        State('annotation_canvas', 'figure'),
        # State('annotation_canvas', 'relayoutData'),
        State('data-collection', 'value'),
        State('uploaded_dict', 'data'),
        Input('image_layers', 'value'),
        State('annotation_canvas', 'style'),
        prevent_initial_call=True)
    def update_canvas_size(value, current_canvas, data_selection,
                           image_dict, add_layer, cur_sizing):
        if None not in (add_layer, value, data_selection, image_dict):
            try:
                first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
                aspect_ratio = int(first_image.shape[1]) / int(first_image.shape[0])
            except (KeyError, AttributeError):
                aspect_ratio = 1

            # if the current canvas is not None, update using the aspect ratio
            # otherwise, use aspect of 1
            if aspect_ratio is None and current_canvas is not None and \
                    'range' in current_canvas['layout']['xaxis'] and \
                    'range' in current_canvas['layout']['yaxis']:
                try:
                    # aspect ratio is width divided by height
                    if aspect_ratio is None:
                        aspect_ratio = int(current_canvas['layout']['xaxis']['range'][1]) / \
                               int(current_canvas['layout']['yaxis']['range'][0])
                except (KeyError, ZeroDivisionError):
                    aspect_ratio = 1

            width = float(value * aspect_ratio)
            height = float(value)
            try:
                if cur_sizing['height'] != f'{height}vh' and cur_sizing['width'] != f'{width}vh':
                    return {'width': f'{width}vh', 'height': f'{height}vh'}
                else:
                    raise PreventUpdate
            except KeyError:
                return {'width': f'{width}vh', 'height': f'{height}vh'}
        # elif value is not None and current_canvas is None:
        #     return {'width': f'{value}vh', 'height': f'{value}vh'}
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
    # @cache.memoize())
    def update_area_information(graph, graph_layout, upload, layers, data_selection, aliases_dict, nclicks,
                                stats_table_open):
        if graph is not None and graph_layout is not None and data_selection is not None and nclicks and stats_table_open:
            return RegionSummary(graph_layout, upload, layers, data_selection, aliases_dict).get_summary_frame()
        elif stats_table_open:
            return pd.DataFrame(
                {'Channel': [], 'Mean': [], 'Max': [], 'Min': [], 'Total': []}).to_dict(orient='records')
        raise PreventUpdate

    @dash_app.callback(Output('image-gallery-row', 'children'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       # Input('image-analysis', 'value'),
                       Input('uploaded_dict', 'data'),
                       Input('data-collection', 'value'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('toggle-gallery-zoom', 'value'),
                       State('preset-options', 'value'),
                       State('image_presets', 'data'),
                       Input('toggle-gallery-view', 'value'),
                       Input('unique-channel-list', 'value'),
                       Input('alias-dict', 'data'),
                       State('preset-button', 'n_clicks'),
                       State('blending_colours', 'data'),
                       Input('default-scaling-gallery', 'value'),
                       State('pixel-level-analysis', 'active_tab'),
                       State('session_config', 'data'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def create_channel_tile_gallery_grid(gallery_data, data_selection, canvas_layout, toggle_gallery_zoom,
                          preset_selection, preset_dict, view_by_channel, channel_selected, aliases, nclicks,
                          blend_colour_dict, toggle_scaling_gallery, active_tab, session_config, delimiter):
        """
        Create a tiled image gallery of the current ROI. If the current dataset selection does not yet have
        default percentile scaling applied, apply before rendering
        IMPORTANT: do not return the blend dictionary here as it will override
        the session blend dictionary on an ROI change
        """
        try:
            # condition 1: if the data collection is changed, update with new images
            # condition 2: if any other mods are made, ensure that the active tab is the gallery tab
            new_collection = gallery_data is not None and ctx.triggered_id in ["data-collection", "uploaded_dict", "alias-dict"]
            gallery_mod_in_tab = gallery_data is not None and ctx.triggered_id not in \
                          ["data-collection", "uploaded_dict", "annotation_canvas"] and active_tab == 'gallery-tab'
            use_zoom = gallery_data is not None and ctx.triggered_id == 'annotation_canvas'
            zoom_not_needed = ctx.triggered_id == 'annotation_canvas' and not toggle_gallery_zoom
            if (new_collection or gallery_mod_in_tab or use_zoom) and not zoom_not_needed:
                zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
                # maintain the original order of channels that is dictated by the metadata
                # decide if channel view or ROI view is selected
                # channel view
                blend_return = dash.no_update
                if view_by_channel and channel_selected is not None:
                    # TODO: modify single-channel view with lazy loading
                    # views = get_all_images_by_channel_name(gallery_data, channel_selected)
                    views = RegionThumbnail(session_config, blend_colour_dict, [channel_selected], 10000,
                                            delimiter=delimiter, use_greyscale=True).get_image_dict()
                    if toggle_scaling_gallery:
                        try:
                            blend_colour_dict = check_blend_dictionary_for_blank_bounds_by_channel(
                                blend_colour_dict, channel_selected, gallery_data, data_selection)
                            views = {key: apply_preset_to_array(resize_for_canvas(value),
                                    blend_colour_dict[channel_selected]) for key, value in views.items()}
                        except KeyError:
                            pass
                else:
                    views = {elem: gallery_data[data_selection][elem] for elem in list(aliases.keys())}

                if views is not None:
                    row_children = generate_channel_tile_gallery_children(views, canvas_layout, zoom_keys, blend_colour_dict,
                                    preset_selection, preset_dict, aliases, nclicks, toggle_gallery_zoom, toggle_scaling_gallery)
                else:
                    row_children = []
                return row_children, blend_return
            raise PreventUpdate
        except (dash.exceptions.LongCallbackError, AttributeError, KeyError):
            raise PreventUpdate

    @dash_app.server.route("/" + str(tmpdirname) + "/" + str(authentic_id) + '/downloads/<path:path>')
    # @cache.memoize())
    def serve_static(path):
        return flask.send_from_directory(
            os.path.join(tmpdirname, str(authentic_id), 'downloads'), path, as_attachment=True)

    @dash_app.callback(Output('blend-options-ag-grid', 'rowData'),
                       Output('blend-options-ag-grid', 'defaultColDef'),
                       Input('blending_colours', 'data'),
                       Input('channel-order', 'data'),
                       State('data-collection', 'value'),
                       Input('alias-dict', 'data'))
    # @cache.memoize())
    def create_ag_grid_legend(blend_colours, current_blend, data_selection, aliases):
        """
        Set the inputs and parameters for the dash ag grid containing the current blend channels
        """
        if current_blend is not None and len(current_blend) > 0:
            in_blend = [aliases[elem] for elem in current_blend]
            cell_styling_conditions = ag_grid_cell_styling_conditions(blend_colours, current_blend, data_selection, aliases)
            if len(in_blend) > 0 and len(cell_styling_conditions) > 0:
                to_return = pd.DataFrame(in_blend, columns=["Channel"]).to_dict(orient="records")
                return to_return, {"sortable": False, "filter": False,
                                   "cellStyle": {"styleConditions": cell_styling_conditions}}
            else:
                return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), {"sortable": False, "filter": False}
        return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), {"sortable": False, "filter": False}

    @dash_app.callback(
        Output("area-stats-collapse", "is_open", allow_duplicate=True),
        [Input("compute-region-statistics", "n_clicks")],
        [State("area-stats-collapse", "is_open")])
    # @cache.memoize())
    def toggle_area_stats_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @dash_app.callback(Output("pixel-hist", 'figure', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'value', allow_duplicate=True),
                       Input('data-collection', 'value'),
                       State('images_in_blend', 'value'),
                       prevent_initial_call=True)
    def reset_pixel_adjustments_on_new_dataset(new_selection, currently_in_blend):
        """
        Reset the pixel histogram and range slider on a new dataset selection
        """
        if currently_in_blend is not None:
            fig = go.Figure()
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis=XAxis(showticklabels=False),
                          yaxis=YAxis(showticklabels=False),
                          margin=dict(l=5, r=5, b=15, t=20, pad=0))
            if new_selection is not None:
                return fig, [None, None]
            else:
                return fig, [None, None]
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
        if blend is None or len(blend) == 0 and len(current_selection) > 0:
            fig = go.Figure()
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis=XAxis(showticklabels=False),
                          yaxis=YAxis(showticklabels=False), margin=dict(l=5, r=5, b=15, t=20, pad=0))
            cur_canvas['data'] = []
            return fig, cur_canvas, [None, None], [], None
        raise PreventUpdate

    @dash_app.callback(Input('images_in_blend', 'value'),
                       Output('custom-slider-max', 'value'),
                       prevent_initial_call=True)
    def reset_range_max_on_channel_switch(new_image_mod):
        """
        Reset the checkbox for a custom range slider max on channel changing. Prevents the slider bar from
        having incorrect bounds for the upcoming channel
        """
        if new_image_mod is not None:
            return []
        raise PreventUpdate

    @dash_app.callback(Output("pixel-hist", 'figure'),
                       Output('pixel-intensity-slider', 'max'),
                       Output('pixel-intensity-slider', 'value'),
                       Output('pixel-intensity-slider', 'marks'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('pixel-intensity-slider', 'step'),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input("pixel-hist-collapse", "is_open"),
                       State('pixel-intensity-slider', 'value'),
                       Input('custom-slider-max', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def update_pixel_histogram_and_intensity_sliders(selected_channel, uploaded, data_selection,
                                    current_blend_dict, show_pixel_hist, cur_slider_values, custom_max):
        """
        Create pixel histogram and output the default percentiles
        """
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict):
            blend_return = dash.no_update
            try:
                if show_pixel_hist and ctx.triggered_id in ["pixel-hist-collapse", "images_in_blend"]:
                    fig, hist_max = pixel_hist_from_array(uploaded[data_selection][selected_channel])
                    fig.update_layout(showlegend=False, yaxis={'title': None},
                                      xaxis={'title': None}, margin=dict(pad=0))
                else:
                    fig = dash.no_update
                    hist_max = float(np.max(uploaded[data_selection][selected_channel]))
            except (ValueError, TypeError):
                fig = dash.no_update
                hist_max = 100.0
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
                        upper_bound = get_default_channel_upper_bound_by_percentile(
                            uploaded[data_selection][selected_channel])
                        current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                        current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                        blend_return = current_blend_dict
                    # if the upper bound is larger than the custom percentile, set it to the upper bound
                    if ' Set range max to current upper bound' in custom_max:
                        hist_max = upper_bound
                        tick_markers, step_size = set_range_slider_tick_markers(hist_max)
                    # set tick spacing between marks on the rangeslider
                    # have 4 tick markers
                    return fig, hist_max, [lower_bound, upper_bound], tick_markers, blend_return, step_size
                except (KeyError, ValueError):
                    return {}, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif ctx.triggered_id == 'blending_colours':
                vals_return = dash.no_update
                if current_blend_dict[selected_channel]['x_lower_bound'] is not None and \
                        current_blend_dict[selected_channel]['x_upper_bound'] is not None:
                    if float(current_blend_dict[selected_channel]['x_lower_bound']) != float(cur_slider_values[0]) or \
                        float(current_blend_dict[selected_channel]['x_upper_bound']) != float(cur_slider_values[1]):
                        lower_bound = float(current_blend_dict[selected_channel]['x_lower_bound'])
                        upper_bound = float(current_blend_dict[selected_channel]['x_upper_bound'])
                        vals_return = [lower_bound, upper_bound]
                else:
                    lower_bound = 0
                    upper_bound = get_default_channel_upper_bound_by_percentile(
                        uploaded[data_selection][selected_channel])
                    current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                    current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                    blend_return = current_blend_dict
                    vals_return = [lower_bound, upper_bound]
                return dash.no_update, hist_max, vals_return, tick_markers, blend_return, step_size
            elif ctx.triggered_id == "pixel-hist-collapse":
                return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
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
                    return dash.no_update, hist_max, cur_slider_values, tick_markers, dash.no_update, step_size
                except IndexError:
                    raise PreventUpdate
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
    # @cache.memoize())
    def reset_intensity_slider_to_default(selected_channel, uploaded, data_selection,
                                                     current_blend_dict, cur_slider_values, reset, cur_max):
        """
        Reset the range slider for the current channel to the default values (min of 0 and max of 99th pixel
        percentile)
        """
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict) and reset > 0:
            hist_max = upper_bound_for_range_slider(uploaded[data_selection][selected_channel])
            upper_bound = float(get_default_channel_upper_bound_by_percentile(uploaded[data_selection][selected_channel]))
            if int(cur_slider_values[0]) != 0 or (int(cur_slider_values[1]) != upper_bound):
                vals_return = [0, upper_bound]
                tick_markers, step_size = set_range_slider_tick_markers(hist_max)
                return hist_max, vals_return, tick_markers, step_size, []
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
                       State("annotation-color-picker", 'value'))
    # @cache.memoize())
    def update_channel_filter_inputs(selected_channel, uploaded, data_selection, current_blend_dict,
                                     preset_selection, preset_dict, session_vars, cur_bool_filter, cur_filter_type,
                                     cur_filter_val, cur_filter_sigma, cur_colour):
        """
        Update the input widgets wth the correct channel configs when the channel is changed, or a preset is used,
        or if the blend dict is updated
        """
        only_options_changed = False
        if None not in (ctx.triggered, session_vars):
            # do not update if the trigger is the channel options and the current selection hasn't changed
            only_options_changed = ctx.triggered_id == "images_in_blend" and \
                                   ctx.triggered[0]['value'] == session_vars["cur_channel"]

        if None not in (selected_channel, uploaded, data_selection, current_blend_dict) and \
                ctx.triggered_id in ["images_in_blend", "blending_colours"] and not only_options_changed:
            filter_type, filter_val, filter_sigma, color = return_current_channel_blend_params(current_blend_dict, selected_channel)
            to_apply_filter = return_current_or_default_filter_apply(cur_bool_filter, filter_type, filter_val, filter_sigma)
            filter_type_return = return_current_or_default_filter_param(cur_filter_type, filter_type)
            filter_val_return = return_current_or_default_filter_param(cur_filter_val, filter_val)
            filter_sigma_return = return_current_or_default_filter_param(cur_filter_sigma, filter_sigma)
            color_return = return_current_or_default_channel_color(cur_colour, color)
            return to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return
        if ctx.triggered_id in ['preset-options'] and None not in \
                (preset_selection, preset_dict, selected_channel, data_selection, current_blend_dict):
            filter_type, filter_val, filter_sigma, color = return_current_channel_blend_params(current_blend_dict,
                                                                                               selected_channel)
            to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return = \
                return_current_default_params_with_preset(filter_type, filter_val, filter_sigma, color)
            return to_apply_filter, filter_type_return, filter_val_return, filter_sigma_return, color_return
        raise PreventUpdate

    @dash_app.callback(Output('sigma-val-filter', 'disabled'),
                       Input('filter-type', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def update_channel_filter_inputs(filter_type):
        return True if filter_type == "median" else False

    @dash_app.callback(Output('global-sigma-val-filter', 'disabled'),
                       Input('global-filter-type', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
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
    # @cache.memoize())
    def generate_preset_options(selected_click, preset_name, current_preset_options, data_selection, layer,
                                current_blend_dict, current_presets, cur_preset_chosen):
        if selected_click is not None and selected_click > 0 and None not in (preset_name, data_selection, layer,
                                                                              current_blend_dict):


            if preset_name not in current_preset_options:
                current_preset_options.append(preset_name)

            current_presets = {} if current_presets is None else current_presets

            current_presets[preset_name] = current_blend_dict[layer]

            if cur_preset_chosen in current_preset_options:
                set_preset = cur_preset_chosen
            else:
                set_preset = None

            return current_preset_options, current_presets, set_preset
        raise PreventUpdate

    @dash_app.callback(Input('image_presets', 'data'),
                       Output('hover-preset-information', 'children'))
    # @cache.memoize())
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
    # @cache.memoize())
    def populate_gallery_channel_list(session_config, aliases):
        """
        Populate a list of all unique channel names for the gallery view
        """
        if session_config is not None and 'unique_images' in session_config.keys():
            try:
                if not all([elem in aliases.keys() for elem in session_config['unique_images']]): raise AssertionError
                return [{'label': aliases[i], 'value': i} for i in session_config['unique_images']]
            except AttributeError:
                # TODO: raise exception on None if the data could not be imported, likely due to a disk storage error
                raise DataImportError(ALERT.warnings["possible-disk-storage-error"])
            except KeyError:
                return []
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
            try:
                cur_vars["cur_channel"] = selected_channel
                return cur_vars
            except KeyError:
                return cur_vars
        raise PreventUpdate

    @dash_app.callback(
        Output("fullscreen-canvas", "is_open"),
        Input('make-canvas-fullscreen', 'n_clicks'),
        [State("fullscreen-canvas", "is_open")])
    def toggle_fullscreen_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(Output('annotation_canvas-fullscreen', 'figure'),
                       Output('annotation_canvas-fullscreen', 'relayoutData'),
                       Input('make-canvas-fullscreen', 'n_clicks'),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas-fullscreen', 'relayoutData'),
                       prevent_initial_call=True)
    def render_canvas_fullscreen(clicks, cur_canvas, cur_layout):
        if clicks > 0 and None not in (cur_layout, cur_canvas):
            fullscreen = FullScreenCanvas(cur_canvas, cur_layout)
            fig = go.Figure(fullscreen.get_canvas())
            fig.update_layout(dragmode='pan')
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis=XAxis(showticklabels=False),
                yaxis=YAxis(showticklabels=False), margin=dict(l=0, r=0, b=0, t=0, pad=0))
            return fig, fullscreen.get_canvas_layout()
        return {}, None

    @dash_app.callback(State('annotation_canvas', 'figure'),
                       Input('annotation_canvas', 'relayoutData'),
                       Output('bound-shower', 'children'),
                       Output('window_config', 'data'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def update_bound_display(cur_graph, cur_graph_layout):
        bound_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
        if None not in (cur_graph, cur_graph_layout) and all([elem in cur_graph_layout for elem in bound_keys]):
            # only update if these keys are used for drag or pan to set custom coords
            x_low = float(min(cur_graph_layout['xaxis.range[0]'], cur_graph_layout['xaxis.range[1]']))
            x_high = float(max(cur_graph_layout['xaxis.range[0]'], cur_graph_layout['xaxis.range[1]']))
            y_low = float(min(cur_graph_layout['yaxis.range[0]'], cur_graph_layout['yaxis.range[1]']))
            y_high = float(max(cur_graph_layout['yaxis.range[0]'], cur_graph_layout['yaxis.range[1]']))
            return html.H6(f"Current bounds: \n X: ({round(x_low, 2)}, {round(x_high, 2)}),"
                           f" Y: ({round(y_low, 2)}, {round(y_high, 2)})",
                           style={"color": "black", "white-space": "pre"}), \
                {"x_low": x_low, "x_high": x_high, "y_low": y_low, "y_high": y_high}
        # if the zoom is reset to the default, clear the bound window
        elif cur_graph_layout in [{'xaxis.autorange': True, 'yaxis.autorange': True}, {'autosize': True}]:
            return [], {"x_low": None, "x_high": None, "y_low": None, "y_high": None}
        # otherwise, keep the bound window (i.e. if a shape is created)
        raise PreventUpdate

    @dash_app.callback(
        Output("dataset-preview", "is_open"),
        Input('show-dataset-info', 'n_clicks'),
        [State("dataset-preview", "is_open")])
    def toggle_dataset_info_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(
        Output("data-collection", "value", allow_duplicate=True),
        Input('dataset-preview-table', 'selected_rows'),
        State('data-collection', 'options'),
        State('data-collection', 'value'),
        prevent_initial_call=True)
    def select_roi_from_preview_table(active_selection, dataset_options, cur_selection):
        if None not in (active_selection, dataset_options) and len(active_selection) > 0:
            try:
                # if the selection is the current one, do nothing
                to_return = dataset_options[active_selection[0]] if \
                    dataset_options[active_selection[0]] != cur_selection else dash.no_update
                return to_return
            except KeyError:
                raise PreventUpdate
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
            except KeyError:
                raise PreventUpdate
        raise PreventUpdate


    @dash_app.callback(
        Output("alert-modal", "is_open"),
        Output("alert-information", "children"),
        Input('session_alert_config', 'data'),
        prevent_initial_call=True)
    def show_alert_modal(alert_dict):
        """
        If the alert dict is populated with a warning, show the warning in the modal. Otherwise, do not populate and
        don't show the modal
        """

        if alert_dict is not None and len(alert_dict) > 0 and "error" in alert_dict.keys() and \
                alert_dict["error"] is not None:
            children = [html.H6("Message: \n"), html.H6(alert_dict["error"])]
            return True, children
        return False, None

    @dash_app.callback(Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('alias-dict', 'data'),
                       Input("imc-metadata-editable", "data"),
                       State('session_alert_config', 'data'),
                       prevent_initial_call=True)
    def give_alert_on_improper_edited_metadata(gene_aliases, metadata_editable, error_config):
        """
        Send an alert when the format of the editable metadata table looks incorrect
        Will arise if more labels are provided than there are channels, which will create blank key entries
        in the metadata list and alias dictionary
        """
        bad_entries = ['', ' ']
        if any([elem in bad_entries for elem in gene_aliases.keys()]) or \
            any([elem['Channel Name'] in bad_entries or elem['Channel Label'] in bad_entries for \
                 elem in metadata_editable]):
            error_config = {"error": None} if error_config is None else error_config
            error_config["error"] = ALERT.warnings["metadata_format_error"]
            return error_config
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
        if None not in (cur_graph_layout, data_selection, current_blend) and len(cur_graph_layout) > 0 and \
                len(current_blend) > 0:
            zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
            if all([elem in cur_graph_layout for elem in zoom_keys]) or 'shapes' in cur_graph_layout and \
                    len(cur_graph_layout['shapes']) > 0:
                return False
            else:
                return True
        return True

    @dash_app.callback(
        Output("region-annotation-modal", "is_open"),
        Input('region-annotation', 'n_clicks'),
         Input('create-annotation', 'n_clicks'))
    def toggle_region_annotation_modal(clicks_add_annotation, clicks_submit_annotation):
        if clicks_add_annotation and ctx.triggered_id == "region-annotation":
            return True
        elif ctx.triggered_id == "create-annotation" and clicks_submit_annotation:
            return False
        else:
            return False

    @dash_app.callback(
        Output("annotations-dict", "data"),
        Input('create-annotation', 'n_clicks'),
        State('region-annotation-name', 'value'),
        State('region-annotation-body', 'value'),
        State('region-annotation-cell-types', 'value'),
        State('annotation_canvas', 'relayoutData'),
        State("annotations-dict", "data"),
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
        State('gating-cell-list', 'data'))
    def add_annotation_to_dict(create_annotation, annotation_title, annotation_body, annotation_cell_type,
                               canvas_layout, annotations_dict, data_selection, cur_layers, mask_toggle,
        mask_selection, mask_blending_level, add_mask_boundary, annot_col, add_annot_gating, apply_gating,
                               gating_annot_col, gating_annot_type, gating_cell_id_list):
        annotations_dict = check_for_valid_annotation_hash(annotations_dict, data_selection)
        # Option 1: if triggered from gating
        if ctx.triggered_id == "gating-annotation-create" and add_annot_gating and apply_gating and None not in \
                (gating_annot_col, gating_annot_type, gating_cell_id_list, mask_selection, data_selection, cur_layers):
            annotations_dict[data_selection][tuple(gating_cell_id_list)] = RegionAnnotation(
                title=None, body=None, cell_type=gating_annot_type, imported=False, annotation_column=gating_annot_col,
                type="gate", channels=cur_layers, use_mask=mask_toggle, mask_selection=mask_selection,
                mask_blending_level=mask_blending_level, add_mask_boundary=add_mask_boundary, id=str(shortuuid.uuid())).dict()
            return SessionServerside(annotations_dict, key="annotation_dict", use_unique_key=app_config['serverside_overwrite'])
        # Option 2: if triggered from region drawing
        elif ctx.triggered_id == "create-annotation" and create_annotation and None not in \
                (annotation_title, annotation_body, canvas_layout, data_selection, cur_layers):
            # use the data collection as the highest key for each ROI, then use the canvas coordinates to
            # uniquely identify a region
            # IMP: convert the dictionary to a sorted tuple to use as a key
            # https://stackoverflow.com/questions/1600591/using-a-python-dictionary-as-a-key-non-nested
            # TODO: add in logic here to allow gated cell annotation from the relevant inputs
            annotation_list = {}
            # Option 1: if zoom is used
            if isinstance(canvas_layout, dict) and 'shapes' not in canvas_layout:
                annotation_list[tuple(sorted(canvas_layout.items()))] = "zoom"
            # Option 2: if a shape is drawn on the canvas
            elif 'shapes' in canvas_layout and isinstance(canvas_layout, dict) and len(canvas_layout['shapes']) > 0:
                # only get the shapes that are a rect or path, the others are canvas annotations
                # set using only the most recent shape to be added to avoid duplication
                for shape in [canvas_layout['shapes'][-1]]:
                    if shape['type'] == 'path':
                        annotation_list[shape['path']] = 'path'
                    elif shape['type'] == "rect":
                        key = {k: shape[k] for k in ('x0', 'x1', 'y0', 'y1')}
                        annotation_list[tuple(sorted(key.items()))] = "rect"
            for key, value in annotation_list.items():
                annotations_dict[data_selection][key] = RegionAnnotation(title = annotation_title, body = annotation_body,
                cell_type = annotation_cell_type, imported = False, annotation_column = annot_col, type = value,
                channels = cur_layers, use_mask = mask_toggle, mask_selection = mask_selection,
                mask_blending_level = mask_blending_level, add_mask_boundary = add_mask_boundary, id=str(shortuuid.uuid())).dict()
            return SessionServerside(annotations_dict, key="annotation_dict", use_unique_key=app_config['serverside_overwrite'])
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
    def toggle_offcanvas_inputs(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(
        Output("blend-config-offcanvas", "is_open"),
        Input("blend-offcanvas-button", "n_clicks"),
        State("blend-config-offcanvas", "is_open"),
    )
    def toggle_offcanvas_blend_options(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(
        Output("pixel-hist-collapse", "is_open", allow_duplicate=True),
        [Input("show-pixel-hist", "n_clicks")],
        [State("pixel-hist-collapse", "is_open")])
    # @cache.memoize())
    def toggle_pixel_hist_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    # @dash_app.callback(Output('canvas-div-holder', 'children', allow_duplicate=True),
    #                    Input('channel-intensity-hover', 'value'),
    #                    Input('add-cell-id-mask-hover', 'value'),
    #                    prevent_initial_call=True)
    # def wrap_canvas_when_using_hovertext(show_channel_intensity_hover, show_mask_hover):
    #     """
    #     Wrap the canvas in a loading screen if hover text is used, as it severely slows down the speed of callbacks
    #     """
    #     if ' Show mask ID on hover' in show_mask_hover or \
    #             " Show channel intensities on hover" in show_channel_intensity_hover:
    #         return [wrap_canvas_in_loading_screen_for_large_images(image=None, hovertext=True)]
    #     else:
    #         return [wrap_canvas_in_loading_screen_for_large_images(image=None, hovertext=False)]

    @dash_app.callback(
        Output("annotation-preview", "is_open"),
        Input('show-annotation-table', 'n_clicks'),
        [State("annotation-preview", "is_open")])
    def toggle_annotation_table_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(
        Output("quant-annotation-col", "options"),
        Output('quant-annotation-col-in-tab', 'options'),
        Output('quant-annotation-col-gating', 'options'),
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
        Will add the category to all of the dropdown menus where annotation can occur: Currently these are:
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
            if col_to_add not in cur_cols:
                cur_cols.append(col_to_add)
            return cur_cols, cur_cols, cur_cols
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
        State('quant-annotation-col', 'value'),
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
                if annotations_dict is None or len(annotations_dict) < 1:
                    annotations_dict = {}
                if data_selection not in annotations_dict.keys():
                    annotations_dict[data_selection] = {}

                x = clickdata['points'][0]['x']
                y = clickdata['points'][0]['y']

                annotations_dict[data_selection][str(clickdata)] = RegionAnnotation(cell_type=annotation_cell_type,
                                            annotation_column=annot_col, type='point', id=str(shortuuid.uuid())).dict()
                if ' Add circle on click' in add_circle:
                    circle_size = int(circle_size)
                    fig = CanvasLayout(cur_figure).clear_improper_shapes()
                    fig['layout']['shapes'].append(
                        {'editable': True, 'line': {'color': 'white'}, 'type': 'circle',
                         'x0': (x - circle_size), 'x1': (x + circle_size),
                         'xref': 'x', 'y0': (y - circle_size), 'y1': (y + circle_size), 'yref': 'y'})
                else:
                    fig = dash.no_update
                return SessionServerside(annotations_dict, key="annotation_dict"), html.H6(f"Point {x, y} updated with "
                                                 f"{annotation_cell_type} in {annot_col}"), True, fig
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
        Render a circle for every valid point annotation imported from a CSV. Valiad xy coordinates
        must fit inside the dimensions of the current image
        """
        if None not in (imported_annotations, image_dict, data_selection, cur_graph):
            first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
            fig = CanvasLayout(cur_graph).add_point_annotations_as_circles(imported_annotations, first_image, circle_size)
            return CanvasLayout(fig).clear_improper_shapes()
        raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'value', allow_duplicate=True),
                       Input('prev-roi', 'n_clicks'),
                       Input('next-roi', 'n_clicks'),
                       Input('keyboard-listener', 'event'),
                       Input('keyboard-listener', 'n_events'),
                       State('data-collection', 'value'),
                       State('data-collection', 'options'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def click_to_new_roi(prev_roi, next_roi, key_listener, n_events, cur_data_selection, cur_options):
        """
        Use the forward and backwards buttons to click to a new ROI
        Alternatively, use the directional arrow buttons from an event listener
        """
        if None not in (cur_data_selection, cur_options):
            cur_index = cur_options.index(cur_data_selection)
            try:
                prev_trigger = previous_roi_trigger(ctx.triggered_id, prev_roi, key_listener, n_events)
                next_trigger = next_roi_trigger(ctx.triggered_id, next_roi, key_listener, n_events)
                if prev_trigger and cur_index != 0:
                    return cur_options[cur_index - 1] if cur_options[cur_index - 1] != cur_data_selection else dash.no_update
                elif next_trigger:
                    return cur_options[cur_index + 1] if cur_options[cur_index + 1] != cur_data_selection else dash.no_update
                else:
                    raise PreventUpdate
            except IndexError:
                raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(Output('prev-roi', 'disabled'),
                       Output('next-roi', 'disabled'),
                       Input('data-collection', 'value'),
                       State('data-collection', 'options'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def toggle_roi_click_through_visibility(cur_data_selection, cur_options):
        """
        Toggle the visibility/availability of the previous and next ROi buttons depending on the current ROI selection
        """
        disabled_prev = True if cur_options[0] == cur_data_selection else False
        disabled_next = True if cur_options[-1] == cur_data_selection else False
        return disabled_prev, disabled_next


    @dash_app.callback(
        Output('image_layers', 'value', allow_duplicate=True),
        Output('pixel-level-analysis', 'active_tab'),
        Input({'type': 'gallery-channel', "index": ALL}, "n_clicks"),
        State('image_layers', 'options'),
        State('image_layers', 'value'),
        State('alias-dict', 'data'),
        prevent_initial_call=True)
    # @cache.memoize())
    def add_channel_layer_through_gallery_click(value, layer_options, current_blend, aliases):
        if not all([elem is None for elem in value]) and None not in (layer_options, current_blend, aliases):
            values = [i["value"] for i in layer_options]
            index_from = ctx.triggered_id["index"]
            if index_from in values and index_from not in current_blend:
                current_blend.append(index_from)
                return current_blend, "pixel-analysis"
            else:
                raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output("session-config-modal", "is_open"),
        Input('session-config-modal-button', 'n_clicks'),
        [State("session-config-modal", "is_open")])
    def toggle_general_config_modal(n1, is_open):
        """
        Open the modal for general session variables
        """
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(Output('annotation_canvas', 'config', allow_duplicate=True),
                       Input('enable-canvas-scroll-zoom', 'value'),
                       State('annotation_canvas', 'config'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def toggle_scroll_zoom_on_canvas(enable_zoom, cur_config):
        """
        Toggle the ability to use scroll zoom on the annotation canvas using the input from the
        session configuration modal. Default value is not enabled
        """
        cur_config = cur_config.copy()
        cur_config['scrollZoom'] = enable_zoom
        return cur_config

    # @dash_app.callback(
    #     Output('app_dest', 'href'),
    #     Input('refresh-app', 'n_clicks'))
    # def refresh_and_clear_app(refresh):
    #     """
    #     Open the modal for general session variables
    #     """
    #     if refresh:
    #         remove_ccramic_caches('/tmp/')
    #         return '/ccramic/'
    #     else:
    #         return '/ccramic/'

    @dash_app.callback(Output('tour_component', 'isOpen'),
                       Input('dash-import-tour', 'n_clicks'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def open_tour_guide(activate_tour):
        """
        Toggle the ability to use scroll zoom on the annotation canvas using the input from the
        session configuration modal. Default value is not enabled
        """
        if activate_tour: return True
        return False
