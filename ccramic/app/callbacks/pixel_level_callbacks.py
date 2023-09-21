import os.path

import dash.exceptions
import dash_bootstrap_components as dbc
import dash_uploader as du
import flask
import pandas as pd
from dash import ctx
from dash_extensions.enrich import Output, Input, State, html
from numpy.core._exceptions import _ArrayMemoryError
from tifffile import imwrite

from ..inputs.pixel_level_inputs import *
from ..parsers.pixel_level_parsers import *
from ..utils.cell_level_utils import *
from pathlib import Path
from plotly.graph_objs.layout import YAxis, XAxis
import json
import pathlib

def init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id):
    """
    Initialize the callbacks associated with pixel level analysis/raw image preprocessing (image loading,
    blending, filtering, scaling, etc.)
    """
    dash_app.config.suppress_callback_exceptions = True

    @du.callback(Output('uploads', 'data'),
                 id='upload-image')
    # @cache.memoize())
    def get_filenames_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if filenames and float(status.progress) == 1.0:
            return filenames
        else:
            raise PreventUpdate

    @du.callback(Output('param_blend_config', 'data', allow_duplicate=True),
                 id='upload-param-json')
    # @cache.memoize())
    def get_param_json_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if filenames and float(status.progress) == 1.0:
            param_json = json.load(open(filenames[0]))
            return param_json
        else:
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
        if path is not None and clicks > 0:
            # TODO: fix ability to read in multiple files at different times
            session_config = cur_session if cur_session is not None and \
                                            len(cur_session['uploads']) > 0 else {'uploads': []}
            if error_config is None:
                error_config = {"error": None}
            # session_config = {'uploads': []}
            if import_type == "filepath":
                if os.path.isfile(path):
                    session_config['uploads'].append(path)
                    error_config["error"] = None
                    return session_config, dash.no_update
                else:
                    error_config["error"] = "Invalid filepath provided. Please verify the following: \n\n" \
                                        "- That the file path provided is a valid local file \n" \
                                        "- If running using Docker or a web version, " \
                                        "local file paths will not be available."
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
                    error_config["error"] = "Invalid directory provided. Please verify the following: \n\n" \
                                            "- That the directory provided exists in the local filesystem \n" \
                                            "- If running using Docker or a web version, " \
                                            "local directories will not be available."
                    return dash.no_update, error_config
            else:
                raise PreventUpdate
        else:
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
                else:
                    dialog.Destroy()
                    raise PreventUpdate
            else:
                dialog.Destroy()
                raise PreventUpdate
        else:
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
        else:
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
                       prevent_initial_call=True)
    def create_upload_dict_from_filepath_string(session_dict, current_blend, error_config):
        """
        Create session variables from the list of imported file paths
        Note that a message will be supplied if more than one type of file is passed
        """
        if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
            unique_suffixes = []
            if error_config is None:
                error_config = {"error": None}
            message = "Read in the following files:\n"
            for upload in session_dict['uploads']:
                suffix = pathlib.Path(upload).suffix
                message = message + f"{upload}\n"
                if suffix not in unique_suffixes:
                    unique_suffixes.append(suffix)
            if len(unique_suffixes) > 1:
                error_config["error"] = "Warning: Multiple different file types were detected on upload. " \
                                        "This may cause problems during analysis. For best performance, " \
                                        "it is recommended to analyze datasets all from the same file type extension " \
                                        "and ensure that all imported datasets share the same panel.\n\n" + message
            else:
                error_config["error"] = message
            upload_dict, blend_dict, unique_images, dataset_information = populate_upload_dict(session_dict['uploads'])
            session_dict['unique_images'] = unique_images
            columns = [{'id': p, 'name': p, 'editable': False} for p in dataset_information.keys()]
            data = pd.DataFrame(dataset_information).to_dict(orient='records')
            blend_return = blend_dict if current_blend is None or len(current_blend) == 0 else dash.no_update
            return Serverside(upload_dict), session_dict, blend_return, columns, data, error_config
        else:
            raise PreventUpdate

    @dash_app.callback(Output('data-collection', 'options'),
                       Output('data-collection', 'value'),
                       Output('image_layers', 'value'),
                       Input('uploaded_dict_template', 'data'),
                       State('data-collection', 'value'),
                       State('image_layers', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def populate_dataset_options(uploaded, cur_data_selection, cur_layers_selected):
        if uploaded is not None:
            datasets = []
            selection_return = None
            channels_return = None
            for roi in uploaded.keys():
                if "metadata" not in roi:
                    datasets.append(roi)
            if cur_data_selection is not None:
                selection_return = cur_data_selection if cur_data_selection in datasets else None
                if cur_layers_selected is not None and len(cur_layers_selected) > 0:
                    channels_return = cur_layers_selected
            return datasets, selection_return, channels_return
        else:
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
            return {}
        else:
            raise PreventUpdate


    @dash_app.callback(Output('image_layers', 'options'),
                       Output('image_layers', 'value', allow_duplicate=True),
                       Output('uploaded_dict', 'data', allow_duplicate=True),
                       Output('canvas-div-holder', 'children'),
                       State('uploaded_dict_template', 'data'),
                       Input('data-collection', 'value'),
                       Input('alias-dict', 'data'),
                       State('image_layers', 'value'),
                       State('session_config', 'data'),
                       prevent_initial_call=True)
    def create_dropdown_options(image_dict, data_selection, names, currently_selected_channels, session_config):
        """
        Update the image layers and dropdown options when a new ROI is selected.
        Additionally, check the dimension of the incoming ROI, and wrap the annotation canvas in a load screen
        if the dimensions are above 3000 for either axis
        """
        # set the default canvas to return without a load screen
        canvas_return = [render_default_annotation_canvas()]
        if image_dict and data_selection:
            # load the specific ROI requested into the dictionary
            # imp: use the channel label for the dropdown view and the name in the background to retrieve
            try:
                image_dict = populate_upload_dict_by_roi(image_dict.copy(), dataset_selection=data_selection,
                                                     session_config=session_config)
                # check if the first image has dimensions greater than 3000. if yes, wrap the canvas in a loader
                if all([image_dict[data_selection][elem] is not None for \
                        elem in image_dict[data_selection].keys()]):
                    # get the first image in the ROI and check the dimensions
                    first_image = list(image_dict[data_selection].keys())[0]
                    first_image = image_dict[data_selection][first_image]
                    canvas_return = [wrap_canvas_in_loading_screen_for_large_images(first_image)]
            except IndexError:
                raise PreventUpdate
            try:
                # if all of the currently selected channels are in the new ROI, keep them. otherwise, reset
                if currently_selected_channels is not None and len(currently_selected_channels) > 0 and \
                    all([elem in image_dict[data_selection].keys() for elem in currently_selected_channels]):
                    channels_selected = list(currently_selected_channels)
                else:
                    channels_selected = []
                return [{'label': names[i], 'value': i} for i in names.keys() if len(i) > 0 and \
                        i not in ['', ' ', None]], channels_selected, Serverside(image_dict), canvas_return
            except AssertionError:
                return [], [], Serverside(image_dict), canvas_return
        else:
            raise PreventUpdate

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
                assert all([elem in names.keys() for elem in chosen_for_blend])
                if chosen_for_blend[-1] != cur_channel_mod:
                    channel_auto_fill = chosen_for_blend[-1]
                else:
                    channel_auto_fill = dash.no_update
                return [{'label': names[i], 'value': i} for i in chosen_for_blend], channel_auto_fill
            except (AssertionError, IndexError):
                raise PreventUpdate
        else:
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
                current_blend_dict = create_new_blending_dict(uploaded)
                return current_blend_dict
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(State('uploaded_dict', 'data'),
                       Input('param_blend_config', 'data'),
                       State('data-collection', 'value'),
                       State('image_layers', 'value'),
                       State('canvas-layers', 'data'),
                       State('blending_colours', 'data'),
                       State('session_alert_config', 'data'),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('image_layers', 'value', allow_duplicate=True),
                       prevent_initial_call=True)
    def update_parameters_from_config_json(uploaded_w_data, new_blend_dict, data_selection,
                                               add_to_layer, all_layers, current_blend_dict, error_config):
        """
        Update the blend layer dictionary and currently selected channels from a JSON upload
        Only applies to the channels that have already been selected: if channels are not in the current blend,
        they will be modified on future selection
        Requires that the channel modification menu be empty to make sure that parameters are updated properly
        """
        if error_config is None:
            error_config = {"error": None}

        if None not in (uploaded_w_data, new_blend_dict, data_selection):
            # conditions where the blend dictionary is updated
            panels_equal = current_blend_dict is not None and len(current_blend_dict) == len(new_blend_dict['channels'])
            match_all = current_blend_dict is None and all([len(uploaded_w_data[roi]) == \
                        len(new_blend_dict['channels']) for roi in uploaded_w_data.keys() if '+++' in roi])
            if panels_equal or match_all:
                current_blend_dict = new_blend_dict['channels'].copy()
                if all_layers is None:
                    all_layers = {data_selection: {}}
                for elem in add_to_layer:
                    # make sure any bounds that are stored as None are overwritten with the default scaling
                    if current_blend_dict[elem]['x_upper_bound'] is None:
                        current_blend_dict[elem]['x_upper_bound'] = \
                        get_default_channel_upper_bound_by_percentile(
                        uploaded_w_data[data_selection][elem])
                    if current_blend_dict[elem]['x_lower_bound'] is None:
                        current_blend_dict[elem]['x_lower_bound'] = 0
                    array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem],
                                                     current_blend_dict[elem])
                    all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                                               current_blend_dict[elem][
                                                                                   'color'])).astype(np.uint8)
                error_config["error"] = "Blend parameters successfully updated from JSON."
                channel_list_return = dash.no_update
                if 'config' in new_blend_dict and 'blend' in new_blend_dict['config'] and all([elem in \
                        current_blend_dict.keys() for elem in new_blend_dict['config']['blend']]):
                    channel_list_return = new_blend_dict['config']['blend']
                return Serverside(all_layers), current_blend_dict, error_config, channel_list_return
            else:
                error_config["error"] = "Error: the blend parameters uploaded from JSON do not " \
                                        "match the current panel length. The update did not occur."
                return dash.no_update, dash.no_update, error_config, dash.no_update
        elif data_selection is None:
            error_config["error"] = "Please select an ROI before importing blend parameters from JSON."
            return dash.no_update, dash.no_update, error_config, dash.no_update
        else:
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
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Output('canvas-layers', 'data', allow_duplicate=True),
                       Output('param_config', 'data', allow_duplicate=True),
                       Output('images_in_blend', 'value', allow_duplicate=True),
                       prevent_initial_call=True)
    def update_blend_dict_on_channel_selection(add_to_layer, uploaded_w_data, current_blend_dict, data_selection,
                                               param_dict, all_layers, preset_selection, preset_dict,
                                               cur_image_in_mod_menu):
        """
        Update the blend dictionary when a new channel is added to the multi-channel selector
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
                    if cur_image_in_mod_menu is not None and cur_image_in_mod_menu in \
                        current_blend_dict.keys():
                        channel_modify = cur_image_in_mod_menu
                else:
                    param_dict["current_roi"] = data_selection
            if all_layers is None:
                all_layers = {}
                all_layers[data_selection] = {}
            for elem in add_to_layer:
                # if the selected channel doesn't have a config yet, create one either from scratch or a preset
                if elem not in current_blend_dict.keys():
                    current_blend_dict[elem] = {'color': None,
                                                                 'x_lower_bound': 0,
                                                                 'x_upper_bound':
                                                                     get_default_channel_upper_bound_by_percentile(
                                                                uploaded_w_data[data_selection][elem]),
                                                                 'filter_type': None,
                                                                 'filter_val': None}
                    current_blend_dict[elem]['color'] = '#FFFFFF'
                    if use_preset_condition:
                        current_blend_dict[elem] = apply_preset_to_blend_dict(
                            current_blend_dict[elem], preset_dict[preset_selection])
                # if the selected channel is in the current blend, check if a preset is used to override
                elif elem in current_blend_dict.keys() and use_preset_condition:
                    # do not override the colour of the curreht channel
                    current_blend_dict[elem] = apply_preset_to_blend_dict(
                        current_blend_dict[elem], preset_dict[preset_selection])
                else:
                    # create a nested dict with the image and all of the filters being used for it
                    # if the same blend parameters have been transferred from another ROI, apply them
                    # set a default upper bound for the channel if the value is None
                    if current_blend_dict[elem]['x_upper_bound'] is None:
                        current_blend_dict[elem]['x_upper_bound'] = \
                        get_default_channel_upper_bound_by_percentile(
                        uploaded_w_data[data_selection][elem])
                    if current_blend_dict[elem]['x_lower_bound'] is None:
                        current_blend_dict[elem]['x_lower_bound'] = 0
                    # TODO: evaluate whether there should be a conditional here if the elem is already
                    #  present in the layers dictionary to save time
                    if elem not in all_layers[data_selection].keys():
                        array_preset = apply_preset_to_array(uploaded_w_data[data_selection][elem],
                                                         current_blend_dict[elem])
                        all_layers[data_selection][elem] = np.array(recolour_greyscale(array_preset,
                                                            current_blend_dict[elem]['color'])).astype(
                        np.uint8)
            return current_blend_dict, Serverside(all_layers), param_dict, channel_modify
        else:
            raise PreventUpdate

    @dash_app.callback(Output("annotation-color-picker", 'value', allow_duplicate=True),
                       Output('swatch-color-picker', 'value'),
                       Input('swatch-color-picker', 'value'),
                       prevent_initial_call=True)
    def update_colour_picker_from_swatch(swatch):
        if swatch is not None:
            return dict(hex=swatch), None
        else:
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
                       State('images_in_blend', 'options'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def update_blend_dict_on_color_selection(colour, layer, uploaded_w_data,
                                    current_blend_dict, data_selection, add_to_layer,
                                    all_layers, filter_chosen, filter_name, filter_value,
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
                        array = filter_by_upper_and_lower_bound(array,
                                                            float(current_blend_dict[layer][
                                                                      'x_lower_bound']),
                                                            float(current_blend_dict[layer][
                                                                      'x_upper_bound']))

                    if len(filter_chosen) > 0 and filter_name is not None:
                        if filter_name == "median":
                            array = median_filter(array, int(filter_value))
                        else:
                            array = gaussian_filter(array, int(filter_value))

                # if filters have been selected, apply them before recolouring


                        # and \
                        # colour['hex'] not in ['#ffffff', '#FFFFFF']:
                    current_blend_dict[layer]['color'] = colour['hex']
                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                                                 colour['hex'])).astype(np.uint8)
                    return current_blend_dict, Serverside(all_layers)
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate
        else:
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
            # do not update if the range values in the slider match the curernt blend params:
            try:
                slider_values = [int(float(elem)) for elem in slider_values]
                lower_bound = min(slider_values)
                upper_bound = max(slider_values)


                if int(float(current_blend_dict[layer]['x_lower_bound'])) == int(float(lower_bound)) and \
                        int(float(current_blend_dict[layer]['x_upper_bound'])) == int(float(upper_bound)):
                    raise PreventUpdate
                else:
                    current_blend_dict[layer]['x_lower_bound'] = int(lower_bound)
                    current_blend_dict[layer]['x_upper_bound'] = int(upper_bound)

                    array = apply_preset_to_array(uploaded_w_data[data_selection][layer],
                                  current_blend_dict[layer])

                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                         current_blend_dict[layer]['color']))

                    return current_blend_dict, Serverside(all_layers)
            except TypeError:
                raise PreventUpdate
        else:
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
        preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
        if None not in (preset_selection, preset_dict, data_selection, current_blend_dict, layer):


            array = uploaded_w_data[data_selection][layer]

            for preset_val in preset_keys:
                current_blend_dict[layer][preset_val] = preset_dict[preset_selection][preset_val]

            array = apply_preset_to_array(array, preset_dict[preset_selection])
            all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                                             current_blend_dict[
                                                                                 layer][
                                                                                 'color']))
            return current_blend_dict, Serverside(all_layers)

        else:
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
                       State("kernel-val-filter", 'value'),
                       State('image_layers', 'value'),
                       State('images_in_blend', 'options'),
                       State('static-session-var', 'data'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def set_blend_options_for_layer_with_bool_filter(layer, uploaded, current_blend_dict, data_selection,
                                                     all_layers, filter_chosen, filter_name, filter_value, cur_layers,
                                                     blend_options, session_vars):

        only_options_changed = False
        if None not in (ctx.triggered, session_vars):
            # do not update if the trigger is the channel options and the current selection hasn't changed
            only_options_changed = ctx.triggered_id == "images_in_blend" and \
                                   ctx.triggered[0]['value'] == session_vars["cur_channel"]

        if None not in (layer, current_blend_dict, data_selection, filter_value, filter_name, all_layers) and \
                not only_options_changed:



            array = uploaded[data_selection][layer]

            # condition where the current inputs are set to not have a filter, and the current blend dict matches
            no_filter_in_both = current_blend_dict[layer]['filter_type'] is None and \
                                current_blend_dict[layer]['filter_val'] is None and \
                                len(filter_chosen) == 0

            # condition where toggling between two channels, and the first one has no filter and the second
            # has a filter. prevent the callback with no actual change
            same_filter_params = current_blend_dict[layer]['filter_type'] == filter_name and \
                                 current_blend_dict[layer]['filter_val'] == filter_value and \
                                 len(filter_chosen) > 0

            if not no_filter_in_both and not same_filter_params:
                # do not update if all of the channels are not in the Channel dict
                blend_options = [elem['value'] for elem in blend_options]
                if all([elem in cur_layers for elem in blend_options]):

                    if current_blend_dict[layer]['x_lower_bound'] is not None and \
                        current_blend_dict[layer]['x_upper_bound'] is not None:
                        array = filter_by_upper_and_lower_bound(array,
                                                            float(current_blend_dict[layer][
                                                                      'x_lower_bound']),
                                                            float(current_blend_dict[layer][
                                                                      'x_upper_bound']))

                    if len(filter_chosen) > 0 and filter_name is not None:
                        if filter_name == "median":
                            array = median_filter(array, int(filter_value))
                        else:
                            array = gaussian_filter(array, int(filter_value))

                        current_blend_dict[layer]['filter_type'] = filter_name
                        current_blend_dict[layer]['filter_val'] = filter_value

                    else:
                        current_blend_dict[layer]['filter_type'] = None
                        current_blend_dict[layer]['filter_val'] = None

                    all_layers[data_selection][layer] = np.array(recolour_greyscale(array,
                                                                                 current_blend_dict[
                                                                                     layer][
                                                                                     'color'])).astype(np.uint8)

                    return current_blend_dict, Serverside(all_layers)
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(
        Input('data-collection', 'value'),
        Output('canvas-layers', 'data', allow_duplicate=True),
        Output("download-collapse", "is_open"))
    def reset_canvas_layers_on_new_dataset(data_selection):
        """
        Reset the canvas layers dictionary containing the cached images for the current canvas in order to
        retain memory. Should be cleared on a new ROi selection
        """
        if data_selection is not None:
            return None, False
        else:
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
                current_blend_dict[key] = apply_preset_to_blend_dict(value,
                                                                                      preset_dict[preset_selection])
            return current_blend_dict
        else:
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
                       Input('data-collection', 'value'),
                       # State('image_layers', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def clear_canvas_on_new_dataset(new_selection):
        if new_selection is not None:
            return go.Figure()
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation_canvas', 'figure'),
                       # Output('annotation_canvas', 'relayoutData'),
                       Output('current_canvas_image', 'data'),
                       Input('canvas-layers', 'data'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       Input('alias-dict', 'data'),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       State('uploaded_dict', 'data'),
                       Input('channel-intensity-hover', 'value'),
                       State('canvas-div-holder', 'children'),
                       State('param_config', 'data'),
                       State('mask-dict', 'data'),
                       Input('apply-mask', 'value'),
                       Input('mask-options', 'value'),
                       State('toggle-canvas-legend', 'value'),
                       State('toggle-canvas-scalebar', 'value'),
                       Input('mask-blending-slider', 'value'),
                       Input('add-mask-boundary', 'value'),
                       Input('channel-order', 'data'),
                       State('legend-size-slider', 'value'),
                       Input('add-cell-id-mask-hover', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def render_canvas_from_layer_mask_hover_change(canvas_layers, currently_selected,
                                                data_selection, blend_colour_dict, aliases,
                                                cur_graph, cur_graph_layout, raw_data_dict,
                                                show_each_channel_intensity,
                                                canvas_children, param_dict,
                                                mask_config, mask_toggle, mask_selection, toggle_legend,
                                                toggle_scalebar, mask_blending_level, add_mask_boundary,
                                                channel_order, legend_size, add_cell_id_hover):

        """
        Update the canvas from a layer dictionary update (The cache dictionary containing the modified image layers
        that will be added together to form the canvas
        or
        if a mask is applied to the canvas
        or
        if the hovertemplate is updated (it is faster to recreate the figure rather than trying to remove the
        hovertemplate)
        """
        if canvas_layers is not None and currently_selected is not None and blend_colour_dict is not None and \
                data_selection is not None and len(currently_selected) > 0 and len(canvas_children) > 0 and \
                param_dict["current_roi"] == data_selection and len(channel_order) > 0:


            legend_text = ''
            for image in channel_order:
                # if blend_colour_dict[image]['color'] not in ['#ffffff', '#FFFFFF']:
                label = aliases[image] if aliases is not None and image in aliases.keys() else image
                legend_text = legend_text + f'<span style="color:' \
                                                f'{blend_colour_dict[image]["color"]}"' \
                                                f'>{label}</span><br>'
            try:
                image = sum([canvas_layers[data_selection][elem].astype(np.float32) for \
                             elem in currently_selected if \
                             elem in canvas_layers[data_selection].keys()]).astype(np.float32)
                image = np.clip(image, 0, 255)
                if mask_toggle and None not in (mask_config, mask_selection) and len(mask_config) > 0:
                    if image.shape[0] == mask_config[mask_selection]["array"].shape[0] and \
                            image.shape[1] == mask_config[mask_selection]["array"].shape[1]:
                        # set the mask blending level based on the slider, by default use an equal blend
                        mask_level = float(mask_blending_level / 100) if mask_blending_level is not None else 1
                        image = cv2.addWeighted(image.astype(np.uint8), 1,
                                                mask_config[mask_selection]["array"].astype(np.uint8), mask_level, 0)
                        if add_mask_boundary and mask_config[mask_selection]["boundary"] is not None:
                            # add the border of the mask after converting back to greyscale to derive the conversion
                            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                                    mask_config[mask_selection]["boundary"].astype(np.uint8), 1, 0)

                fig = px.imshow(Image.fromarray(image.astype(np.uint8)))
                # fig.update(data=[{'customdata': )
                image_shape = image.shape
                fig.update_traces(hoverinfo="skip")
                x_axis_placement = 0.00001 * image_shape[1]
                # make sure the placement is min 0.05 and max 0.1
                x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.15 else 0.05
                # if the current graph already has an image, take the existing layout and apply it to the new figure
                # otherwise, set the uirevision for the first time
                # fig = add_scale_value_to_figure(fig, image_shape, x_axis_placement)
                # do not update if there is already a hover template as it will be too slow
                # scalebar is y = 0.06
                # legend is y = 0.05
                hover_template_exists = 'data' in cur_graph and 'customdata' in cur_graph['data'] and \
                                        cur_graph['data']['customdata'] is not None
                if 'layout' in cur_graph and 'uirevision' in cur_graph['layout'] and \
                        cur_graph['layout']['uirevision'] and not hover_template_exists:
                    try:
                        # fig['layout'] = cur_graph['layout']
                        cur_graph['data'] = fig['data']
                        # if taking the old layout, remove the current legend and remake with the new layers
                        # imp: do not remove the current scale bar value if its there
                        if 'annotations' in cur_graph['layout'] and len(cur_graph['layout']['annotations']) > 0:
                            cur_graph['layout']['annotations'] = [annotation for annotation in \
                                                          cur_graph['layout']['annotations'] if \
                                                                  annotation['y'] == 0.06 and toggle_scalebar]
                        if 'shapes' in cur_graph['layout'] and len(cur_graph['layout']['shapes']):
                            cur_graph['layout']['shapes'] = []
                        fig = cur_graph
                        # del cur_graph
                    # keyerror could happen if the canvas is reset with no layers, so rebuild from scratch
                    except KeyError:
                        fig['layout']['uirevision'] = True

                        if toggle_scalebar:
                            fig = add_scale_value_to_figure(fig, image_shape, font_size=legend_size,
                                                            x_axis_left=x_axis_placement)

                        fig = go.Figure(fig)
                        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                          xaxis=XAxis(showticklabels=False, domain=[0, 1]),
                                          yaxis=YAxis(showticklabels=False),
                                          margin=dict(
                                              l=10,
                                              r=0,
                                              b=25,
                                              t=35,
                                              pad=0
                                          ))
                        fig.update_layout(hovermode="x")
                else:
                    # del cur_graph
                    # if making the fig for the firs time, set the uirevision
                    fig['layout']['uirevision'] = True

                    if toggle_scalebar:
                        fig = add_scale_value_to_figure(fig, image_shape, font_size=legend_size,
                                                        x_axis_left=x_axis_placement)

                    fig = go.Figure(fig)
                    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                  xaxis=XAxis(showticklabels=False),
                                  yaxis=YAxis(showticklabels=False),
                                  margin=dict(
                                      l=10,
                                      r=0,
                                      b=25,
                                      t=35,
                                      pad=0
                                  ))
                    fig.update_layout(hovermode="x")

                fig = go.Figure(fig)

                # set how far in from the lefthand corner the scale bar and colour legends should be
                # higher values mean closer to the centre
                # fig = canvas_layers[image_type][currently_selected[0]]
                if legend_text != '' and toggle_legend:
                    fig.add_annotation(text=legend_text, font={"size": legend_size + 1}, xref='paper',
                                           yref='paper',
                                           x=(1 - x_axis_placement),
                                           # xanchor='right',
                                           y=0.05,
                                           # yanchor='bottom',
                                           bgcolor="black",
                                           showarrow=False)

                # set the x-axis scale placement based on the size of the image
                # for adding a scale bar
                if toggle_scalebar:
                    fig.add_shape(type="line",
                                  xref="paper", yref="paper",
                                  x0=x_axis_placement, y0=0.05, x1=(x_axis_placement + 0.075),
                                  y1=0.05,
                                  line=dict(
                                      color="white",
                                      width=2,
                                  ),
                                  )

                # set the custom hovertext if is is requested
                # the masking mask ID get priority over the channel intensity hover
                # TODO: combine both the mask ID and channel intensity into one hover if both are requested

                if mask_toggle and None not in (mask_config, mask_selection) and len(mask_config) > 0 and \
                        ' show mask ID on hover' in add_cell_id_hover:
                    try:
                        # fig.update(data=[{'customdata': None}])
                        fig.update(data=[{'customdata': mask_config[mask_selection]["hover"]}])
                        new_hover = per_channel_intensity_hovertext(["mask ID"])
                    except KeyError:
                        new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"

                elif " show channel intensities on hover" in show_each_channel_intensity:
                    # fig.update(data=[{'customdata': None}])
                    hover_stack = np.stack(tuple(raw_data_dict[data_selection][elem] for elem in currently_selected),
                                           axis=-1)
                    fig.update(data=[{'customdata': hover_stack}])
                    # set the labels for the hover from the aliases
                    hover_labels = []
                    for label in currently_selected:
                        if label in aliases.keys():
                            hover_labels.append(aliases[label])
                        else:
                            hover_labels.append(label)
                    new_hover = per_channel_intensity_hovertext(hover_labels)
                    # fig.update_traces(hovertemplate=new_hover)
                else:
                    fig.update(data=[{'customdata': None}])
                    new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"
                fig.update_traces(hovertemplate=new_hover)
                # fig.update_layout(dragmode="zoom")
                return fig, Serverside(image)
            except (ValueError, AttributeError):
                return dash.no_update
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

        else:
            raise PreventUpdate


    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       Output('annotation_canvas', 'relayoutData', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       Input('annotation_canvas', 'relayoutData'),
                       State('set-x-auto-bound', 'value'),
                       State('set-y-auto-bound', 'value'),
                       State('window_config', 'data'),
                       Input('activate-coord', 'n_clicks'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def render_canvas_from_coord_change(cur_graph, cur_graph_layout, x_request, y_request, current_window,
                             nclicks_coord):

        """
        Update the annotation canvas when the zoom or custom coordinates are requested.
        """

        bad_update = cur_graph_layout in [{"autosize": True}]

        # update the scale bar with and without zooming
        if cur_graph is not None and \
             'shapes' not in cur_graph_layout and cur_graph_layout not in [{'dragmode': 'drawclosedpath'}] and \
                not bad_update:
            if ctx.triggered_id == "annotation_canvas":
                try:
                    fig = go.Figure(cur_graph)
                    # find the text annotation that has um in the text and the correct location
                    for annotations in cur_graph['layout']['annotations']:
                        # if 'μm' in annotations['text'] and annotations['y'] == 0.06:
                        if annotations['y'] == 0.06:
                            if cur_graph_layout not in [{'autosize': True}]:
                                x_range_high = 0
                                x_range_low = 0
                                # use different variables depending on how the ranges are written in the dict
                                # IMP: the variables will be written differently after a tab change
                                if 'xaxis' in cur_graph['layout']:
                                    high = max(cur_graph['layout']['xaxis']['range'][1],
                                           cur_graph['layout']['xaxis']['range'][0])
                                    low = min(cur_graph['layout']['xaxis']['range'][1],
                                          cur_graph['layout']['xaxis']['range'][0])
                                    x_range_high = math.ceil(int(high))
                                    x_range_low = math.floor(int(low))
                                elif 'xaxis.range[0]' and 'xaxis.range[1]' in cur_graph_layout:
                                    high = max(cur_graph_layout['xaxis.range[1]'],
                                           cur_graph_layout['xaxis.range[0]'])
                                    low = min(cur_graph_layout['xaxis.range[1]'],
                                          cur_graph_layout['xaxis.range[0]'])
                                    x_range_high = math.ceil(int(high))
                                    x_range_low = math.ceil(int(low))

                                assert x_range_high >= x_range_low
                                # assert that all values must be above 0 for the scale value to render during panning
                                # assert all([elem >=0 for elem in cur_graph_layout.values() if isinstance(elem, float)])
                                scale_val = int(math.ceil(int(0.075 * (x_range_high - x_range_low))) + 1)
                                scale_val = scale_val if scale_val > 0 else 1
                                scale_annot = str(scale_val) + "μm"
                                scale_text = f'<span style="color: white">{str(scale_annot)}</span><br>'
                                # get the index of the list element corresponding to this text annotation
                                index = cur_graph['layout']['annotations'].index(annotations)
                                cur_graph['layout']['annotations'][index]['text'] = scale_text

                                fig = go.Figure(cur_graph)

                    return fig, cur_graph_layout
                except (ValueError, KeyError, AssertionError):
                    raise PreventUpdate
            if ctx.triggered_id == "activate-coord":
                if None not in (x_request, y_request, current_window) and \
                        nclicks_coord is not None and nclicks_coord > 0:
                    try:
                        # calculate midway distance for each coord. this distance is
                        # added on either side of the x and y requests
                        new_x_low, new_x_high, new_y_low, new_y_high = create_new_coord_bounds(current_window,
                                                                                               x_request,
                                                                                               y_request)
                        new_layout = {'xaxis.range[0]': new_x_low, 'xaxis.range[1]': new_x_high,
                                      'yaxis.range[0]': new_y_high, 'yaxis.range[1]': new_y_low}
                        # IMP: for yaxis, need to set the min and max in the reverse order
                        fig = go.Figure(data=cur_graph['data'], layout=cur_graph['layout'])
                        shapes = cur_graph['layout']['shapes']
                        annotations = cur_graph['layout']['annotations']
                        fig['layout']['shapes'] = None
                        fig['layout']['annotations'] = None
                        fig.update_layout(xaxis=XAxis(showticklabels=False, range=[new_x_low, new_x_high]),
                                          yaxis=YAxis(showticklabels=False, range=[new_y_high, new_y_low]))
                        # cur_graph['layout']['xaxis']['domain'] = [0, 1]
                        # cur_graph['layout']['dragmode'] = "zoom"
                        fig['layout']['shapes'] = shapes
                        fig['layout']['annotations'] = annotations
                        return fig, new_layout
                    except (AssertionError, TypeError):
                        raise PreventUpdate
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate
        else:
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

    @dash_app.callback(Output('annotation_canvas', 'figure', allow_duplicate=True),
                       State('annotation_canvas', 'figure'),
                       State('annotation_canvas', 'relayoutData'),
                       Input('custom-scale-val', 'value'),
                       prevent_initial_call=True)
    # @cache.memoize())
    def render_canvas_from_scalebar_change(cur_graph, cur_graph_layout, custom_scale_val):
        if cur_graph is not None and cur_graph_layout not in [{'dragmode': 'pan'}]:
            try:
                for annotations in cur_graph['layout']['annotations']:
                    # if 'μm' in annotations['text'] and annotations['y'] == 0.06:
                    if annotations['y'] == 0.06:
                        if custom_scale_val is None:
                            high = max(cur_graph['layout']['xaxis']['range'][1],
                                       cur_graph['layout']['xaxis']['range'][0])
                            low = min(cur_graph['layout']['xaxis']['range'][1],
                                      cur_graph['layout']['xaxis']['range'][0])
                            x_range_high = math.ceil(int(high))
                            x_range_low = math.floor(int(low))
                            assert x_range_high >= x_range_low
                            custom_scale_val = int(math.ceil(int(0.075 * (x_range_high - x_range_low))) + 1)
                        scale_annot = str(custom_scale_val) + "μm"
                        scale_text = f'<span style="color: white">{str(scale_annot)}</span><br>'
                        # get the index of the list element corresponding to this text annotation
                        index = cur_graph['layout']['annotations'].index(annotations)
                        cur_graph['layout']['annotations'][index]['text'] = scale_text
                fig = go.Figure(cur_graph)
            except (KeyError, AssertionError):
                fig = dash.no_update
            return fig
        else:
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
                       State('channel-order', 'data'),
                       State('legend-size-slider', 'value'),
                       prevent_initial_call=True)
    def render_canvas_from_toggle_show_annotations(toggle_legend, toggle_scalebar,
                                                   cur_canvas, cur_layout, currently_selected,
                                                   data_selection, blend_colour_dict, aliases, image_dict,
                                                   channel_order, legend_size):
        """
        re-render the canvas if the user requests to remove the annotations (scalebar and legend)
        """
        if None not in (cur_layout, cur_canvas, data_selection, currently_selected, blend_colour_dict):
            # scalebar is y = 0.06
            # legend is y = 0.05


            first_image = list(image_dict[data_selection].keys())[0]
            first_image = image_dict[data_selection][first_image]
            image_shape = first_image.shape
            x_axis_placement = 0.00001 * image_shape[1]
            # make sure the placement is min 0.05 and max 0.1
            x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.15 else 0.05
            if 'layout' in cur_canvas and 'annotations' in cur_canvas['layout']:
                cur_annotations = cur_canvas['layout']['annotations'].copy()
            else:
                cur_annotations = []
            if 'layout' in cur_canvas and 'shapes' in cur_canvas['layout']:
                cur_shapes = cur_canvas['layout']['shapes'].copy()
            else:
                cur_shapes = []
            if ctx.triggered_id == "toggle-canvas-legend":
                if not toggle_legend:
                    cur_annotations = [annot for annot in cur_annotations if \
                                       annot is not None and 'y' in annot and annot['y'] != 0.05]
                    cur_canvas['layout']['annotations'] = cur_annotations
                    return cur_canvas
                else:
                    legend_text = ''
                    # cur_canvas['layout']['shapes'] = [shape for shape in cur_canvas['layout']['shapes'] if \
                    #                                   shape is not None and 'label' in shape and \
                    #                                   shape['label'] is not None and 'texttemplate' not in shape[
                    #                                       'label']]
                    for image in channel_order:
                        # if blend_colour_dict[image]['color'] not in ['#ffffff', '#FFFFFF']:
                        label = aliases[image] if aliases is not None and image in aliases.keys() else image
                        legend_text = legend_text + f'<span style="color:' \
                                                        f'{blend_colour_dict[image]["color"]}"' \
                                                        f'>{label}</span><br>'

                    fig = go.Figure(cur_canvas)
                    if legend_text != '':
                        fig.add_annotation(text=legend_text, font={"size": legend_size + 1}, xref='paper',
                                               yref='paper',
                                               x=(1 - x_axis_placement),
                                               # xanchor='right',
                                               y=0.05,
                                               # yanchor='bottom',
                                               bgcolor="black",
                                               showarrow=False)
                    return fig
            elif ctx.triggered_id == "toggle-canvas-scalebar":
                if not toggle_scalebar:
                    cur_shapes = [shape for shape in cur_shapes if \
                                      shape is not None and 'type' in shape and shape['type'] \
                                      in ['rect', 'path']]
                    cur_annotations = [annot for annot in cur_annotations if \
                                           annot is not None and 'y' in annot and annot['y'] != 0.06]
                    cur_canvas['layout']['annotations'] = cur_annotations
                    cur_canvas['layout']['shapes'] = cur_shapes
                    return cur_canvas
                else:
                    fig = go.Figure(cur_canvas)
                    fig.add_shape(type="line",
                                  xref="paper", yref="paper",
                                  x0=x_axis_placement, y0=0.05, x1=(x_axis_placement + 0.075),
                                  y1=0.05,
                                  line=dict(
                                      color="white",
                                      width=2,
                                  ),
                                  )

                    try:
                        high = max(cur_canvas['layout']['xaxis']['range'][1],
                                   cur_canvas['layout']['xaxis']['range'][0])
                        low = min(cur_canvas['layout']['xaxis']['range'][1],
                                  cur_canvas['layout']['xaxis']['range'][0])
                        x_range_high = math.ceil(int(high))
                        x_range_low = math.floor(int(low))
                        assert x_range_high >= x_range_low
                        custom_scale_val = int(math.ceil(int(0.075 * (x_range_high - x_range_low))) + 1)
                    except KeyError:
                        custom_scale_val = None

                    fig = add_scale_value_to_figure(fig, image_shape, scale_value=custom_scale_val,
                                                    font_size=legend_size, x_axis_left=x_axis_placement)
                return fig
        else:
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
                annotations_copy = cur_graph['layout']['annotations'].copy()
                for annotation in annotations_copy:
                    # the scalebar is always slightly smaller
                    if annotation['y'] == 0.06:
                        annotation['font']['size'] = legend_size
                    elif annotation['y'] == 0.05 and 'color' in annotation['text']:
                        annotation['font']['size'] = legend_size + 1
                cur_graph['layout']['annotations'] = [elem for elem in cur_graph['layout']['annotations'] if \
                                                  elem is not None and 'texttemplate' not in elem]
                return cur_graph
            except KeyError:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @du.callback(Output('metadata_config', 'data'),
                 id='upload-metadata')
    # @cache.memoize())
    def upload_custom_metadata_panel(status: du.UploadStatus):
        """
        Upload a metadata panel separate from the auto-generated metadata panel. This must be parsed against the existing
        datasets to ensure that it matches the number of channels
        """
        filenames = [str(x) for x in status.uploaded_files]
        metadata_config = {'uploads': []}
        if filenames:
            for file in filenames:
                metadata_config['uploads'].append(file)
            return metadata_config
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("imc-metadata-editable", "columns", allow_duplicate=True),
        Output("imc-metadata-editable", "data", allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Input('metadata_config', 'data'),
        State('uploaded_dict_template', 'data'),
        State('session_alert_config', 'data'),
        prevent_initial_call=True)
    # @cache.memoize())
    def populate_datatable_columns(metadata_config, uploaded, error_config):
        if metadata_config is not None and len(metadata_config['uploads']) > 0:
            if error_config is None:
                error_config = {"error": None}
            metadata_read = pd.read_csv(metadata_config['uploads'][0])
            metadata_validated = validate_incoming_metadata_table(metadata_read, uploaded)
            if metadata_validated is not None and 'ccramic Label' not in metadata_validated.keys():
                metadata_validated['ccramic Label'] = metadata_validated["Channel Label"]
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                        p in metadata_validated.keys()], \
                    pd.DataFrame(metadata_validated).to_dict(orient='records'), dash.no_update
            else:
                error_config["error"] = "Could not import custom metadata. Ensure that: \n \n- The dataset " \
                                        "containing the images is " \
                                        "uploaded first" \
                                        "\n - the columns `Channel Name` and " \
                                        "`Channel Label` are present \n - the number of rows matches the number of " \
                                        "channels in the current dataset. \n"
                return dash.no_update, dash.no_update, error_config
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("imc-metadata-editable", "columns"),
        Output("imc-metadata-editable", "data"),
        Input('uploaded_dict_template', 'data'),
        Input('image-metadata', 'data'))
    # @cache.memoize())
    def populate_datatable_columns(uploaded, column_dict):
        if uploaded is not None and uploaded['metadata'] is not None:
            try:
                return [{'id': p, 'name': p, 'editable': make_metadata_column_editable(p)} for
                    p in uploaded['metadata'].keys()], \
                    pd.DataFrame(uploaded['metadata']).to_dict(orient='records')
            except ValueError:
                raise PreventUpdate
        elif column_dict is not None:
            return column_dict["columns"], column_dict["data"]
        else:
            raise PreventUpdate

    @dash_app.callback(
        Input("imc-metadata-editable", "data"),
        Output('alias-dict', 'data'))
    # @cache.memoize())
    def create_channel_label_dict(metadata):
        if metadata is not None:
            alias_dict = {}
            for elem in metadata:
                alias_dict[elem['Channel Name']] = elem['ccramic Label']
            return alias_dict

    @dash_app.callback(
        Output("download-edited-table", "data"),
        Input("btn-download-metadata", "n_clicks"),
        Input("imc-metadata-editable", "data"))
    # @cache.memoize())
    def download_edited_metadata(n_clicks, datatable_contents):
        if n_clicks is not None and n_clicks > 0 and datatable_contents is not None and \
                ctx.triggered_id == "btn-download-metadata":
            return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "metadata.csv")
        else:
            raise PreventUpdate

    @dash_app.callback(Output('download-link', 'href'),
                       Output('download-link-canvas-tiff', 'href'),
                       Output('download-canvas-interactive-html', 'href'),
                       Output('download-blend-config', 'href'),
                       State('uploaded_dict', 'data'),
                       State('imc-metadata-editable', 'data'),
                       State('blending_colours', 'data'),
                       Input("open-download-collapse", "n_clicks"),
                       Input("download-collapse", "is_open"),
                       State('data-collection', 'value'),
                       State('current_canvas_image', 'data'),
                       State('annotation_canvas', 'figure'),
                       State('image_layers', 'value'),
                       State('annotation_canvas', 'style'))
    # @cache.memoize())
    def update_download_href_h5(uploaded, metadata_sheet, blend_dict, nclicks, download_open, data_selection,
                                current_image_tiff, current_canvas, blend_layers, canvas_style):
        """
        Create the download links for the current canvas and the session data.
        Only update if the download dialog is open to avoid continuous updating on canvas change
        """
        if None not in (uploaded, blend_dict) and nclicks > 0 and download_open:

            dest_path = os.path.join(tmpdirname, authentic_id, 'downloads')

            # fig_bytes = pio.to_image(fig, height=image.shape[1], width=image.shape[0])
            # buf = io.BytesIO(fig_bytes)
            # img = Image.open(buf)
            dest_file = dash.no_update
            if current_image_tiff is not None:
                dest_file = str(os.path.join(dest_path, "canvas.tiff"))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                imwrite(dest_file, current_image_tiff.astype(np.uint8), photometric='rgb')

            download_dir = os.path.join(tmpdirname,
                                        authentic_id,
                                        'downloads')
            relative_filename = os.path.join(download_dir, 'data.h5')
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            hf = None
            try:
                hf = h5py.File(relative_filename, 'w')
            except OSError:
                os.remove(relative_filename)
            if hf is None:
                hf = h5py.File(relative_filename, 'w')
            try:
                meta_to_write = pd.DataFrame(metadata_sheet) if metadata_sheet is not None else \
                    pd.DataFrame(uploaded['metadata'])
                for col in meta_to_write:
                    meta_to_write[col] = meta_to_write[col].astype(str)
                hf.create_dataset('metadata', data=meta_to_write.to_numpy())
                hf.create_dataset('metadata_columns', data=meta_to_write.columns.values.astype('S'))
                hf.create_group(data_selection)
                for key, value in uploaded[data_selection].items():
                    if key not in hf[data_selection]:
                        hf[data_selection].create_group(key)
                        if 'image' not in hf[data_selection][key] and value is not None:
                            hf[data_selection][key].create_dataset('image', data=value)
                            if blend_dict is not None and key in blend_dict.keys():
                                for blend_key, blend_val in blend_dict[key].items():
                                    data_write = str(blend_val) if blend_val is not None else "None"
                                    hf[data_selection][key].create_dataset(blend_key, data=data_write)
                            else:
                                pass
                try:
                    hf.close()
                except:
                    pass
                # instead, give a measure at the bottom converting pixels to distance
                # aspect_ratio = float(current_image_tiff.shape[1] / current_image_tiff.shape[0])
                # only change the colour to black if the aspect ratio is not wide, otherwise the image will still
                # fill the HTML so want to keep white
                # scale_update = 'black' if aspect_ratio < 1.75 else 'white'
                # for annotation in current_canvas['layout']['annotations']:
                #     if 'μm' in annotation['text']:
                #         annotation['text'] = f'<span style="color: {scale_update}">1 pixel = 1μm (unzoomed)</span><br>'
                # for shape in current_canvas['layout']['shapes']:
                #     if shape['type'] == 'line' and shape['y0'] == 0.05 and 'line' in shape:
                #         current_canvas['layout']['shapes'].remove(shape)

                # can set the canvas width and height from the ccanvas style to retain the in-app aspect ratio
                fig = go.Figure(current_canvas)
                # fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                #                   xaxis=XAxis(showticklabels=False),
                #                   yaxis=YAxis(showticklabels=False),
                #                   margin=dict(l=0, r=0, b=0, t=0, pad=0))
                fig.write_html(str(os.path.join(download_dir, "canvas.html")), default_width = canvas_style['width'],
                               default_height = canvas_style['height'])
                param_json = str(os.path.join(download_dir, 'param.json'))
                with open(param_json, "w") as outfile:
                    dict_write = {"channels": blend_dict, "config": {"blend": blend_layers}}
                    json.dump(dict_write, outfile)

                return str(relative_filename), dest_file, str(os.path.join(download_dir, "canvas.html")), param_json
            # if the dictionary hasn't updated to include all the experiments, then don't update download just yet
            except KeyError:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(
        Input('current_canvas_image', 'data'),
        Input('annotation_canvas', 'figure'),
        State("download-collapse", "is_open"),
        Output("download-collapse", "is_open", allow_duplicate=True),
        prevent_initial_call=True)
    def reset_canvas_layers_on_new_dataset(current_image, current_canvas, currently_open):
        """
        Close the collapsible download when an update is made to the canvas to prevent extraneous downloading
        """
        if current_canvas is not None:
            if currently_open:
                return False
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate


    @dash_app.callback(
        Output('annotation_canvas', 'style'),
        Input('annotation-canvas-size', 'value'),
        State('annotation_canvas', 'figure'),
        State('annotation_canvas', 'relayoutData'),
        Input('autosize-canvas', 'n_clicks'),
        State('data-collection', 'value'),
        State('uploaded_dict', 'data'),
        Input('image_layers', 'value'),
        State('annotation_canvas', 'style'),
        prevent_initial_call=True)
    def update_canvas_size(value, current_canvas, cur_graph_layout, nclicks, data_selection,
                           image_dict, add_layer, cur_sizing):
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']

        # only update the resolution if not using zoom or panning
        if all([elem not in cur_graph_layout for elem in zoom_keys]) and \
                'dragmode' not in cur_graph_layout.keys() and 'shapes' not in cur_graph_layout.keys() and \
                add_layer is not None and value is not None:
            try:



                first_image = list(image_dict[data_selection].keys())[0]
                first_image = image_dict[data_selection][first_image]
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

            width = value * aspect_ratio
            height = value
            try:
                if cur_sizing['height'] != f'{height}vh' and cur_sizing['width'] != f'{width}vh':
                    return {'width': f'{width}vh', 'height': f'{height}vh'}
                else:
                    raise PreventUpdate
            except KeyError:
                return {'width': f'{width}vh', 'height': f'{height}vh'}
        # elif value is not None and current_canvas is None:
        #     return {'width': f'{value}vh', 'height': f'{value}vh'}
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("selected-area-table", "data"),
        State('annotation_canvas', 'figure'),
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
        # these range keys correspond to the zoom feature
        zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
        # these keys are used if a shape has been created, then modified
        modified_rect_keys = ['shapes[1].x0', 'shapes[1].x1', 'shapes[1].y0', 'shapes[1].y1']

        if graph is not None and graph_layout is not None and data_selection is not None and \
                nclicks and stats_table_open:


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
                                    upload[data_selection][layer],
                                    x_range_low,
                                    x_range_high,
                                    y_range_low, y_range_high)
                                shapes_mean.append(round(float(mean_exp), 2))
                                shapes_max.append(round(float(max_xep), 2))
                                shapes_min.append(round(float(min_exp), 2))
                            # option 2: if a closed form shape is drawn
                            elif shape['type'] == 'path' and 'path' in shape:
                                mean_exp, max_xep, min_exp = get_area_statistics_from_closed_path(
                                    upload[data_selection][layer], shape['path'])
                                shapes_mean.append(round(float(mean_exp), 2))
                                shapes_max.append(round(float(max_xep), 2))
                                shapes_min.append(round(float(min_exp), 2))

                        mean_panel.append(round(sum(shapes_mean) / len(shapes_mean), 2))
                        max_panel.append(round(sum(shapes_max) / len(shapes_max), 2))
                        min_panel.append(round(sum(shapes_min) / len(shapes_min), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    except (AssertionError, ValueError, ZeroDivisionError, IndexError, TypeError,
                            _ArrayMemoryError):
                        pass

                layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}
                return pd.DataFrame(layer_dict).to_dict(orient='records')

            # option 2: if the zoom is used
            elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                    all([elem in graph_layout for elem in zoom_keys]):

                try:
                    assert all([elem >= 0 for elem in graph_layout.keys() if isinstance(elem, float)])
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
                        mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[data_selection][layer],
                                                                                   x_range_low,
                                                                                   x_range_high,
                                                                                   y_range_low, y_range_high)
                        mean_panel.append(round(float(mean_exp), 2))
                        max_panel.append(round(float(max_xep), 2))
                        min_panel.append(round(float(min_exp), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

                    return pd.DataFrame(layer_dict).to_dict(orient='records')

                except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
                    return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                         'Min': []}).to_dict(orient='records')

            # option 3: if a shape has already been created and is modified
            elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                    all([elem in graph_layout for elem in modified_rect_keys]):
                try:
                    assert all([elem >= 0 for elem in graph_layout.keys() if isinstance(elem, float)])
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
                        mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[data_selection][layer],
                                                                                   x_range_low,
                                                                                   x_range_high,
                                                                                   y_range_low, y_range_high)
                        mean_panel.append(round(float(mean_exp), 2))
                        max_panel.append(round(float(max_xep), 2))
                        min_panel.append(round(float(min_exp), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

                    return pd.DataFrame(layer_dict).to_dict(orient='records')

                except (AssertionError, ValueError, ZeroDivisionError, _ArrayMemoryError):
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
                                upload[data_selection][layer], shape_path)
                            mean_panel.append(round(float(mean_exp), 2))
                            max_panel.append(round(float(max_xep), 2))
                            min_panel.append(round(float(min_exp), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

                    layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

                    return pd.DataFrame(layer_dict).to_dict(orient='records')

                except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
                    return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                         'Min': []}).to_dict(orient='records')
            else:
                return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                     'Min': []}).to_dict(orient='records')
        elif stats_table_open:
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')
        else:
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
                       State('unique-channel-list', 'value'),
                       State('alias-dict', 'data'),
                       State('preset-button', 'n_clicks'),
                       State('blending_colours', 'data'),
                       Input('default-scaling-gallery', 'value'),
                       State('pixel-level-analysis', 'active_tab'),
                       prevent_initial_call=True)
    # @cache.memoize()
    def create_image_grid(gallery_data, data_selection, canvas_layout, toggle_gallery_zoom,
                          preset_selection, preset_dict, view_by_channel, channel_selected, aliases, nclicks,
                          blend_colour_dict, toggle_scaling_gallery, active_tab):
        """
        Create a tiled image gallery of the current ROI. If the current dataset selection does not yet have
        default percentile scaling applied, apply before rendering
        IMPORTANT: do not return the blend dictionary here as it will override
        the session blend dictionary on an ROI change
        """
        try:
            # condition 1: if the data collection is changed, update with new images
            # condition 2: if any other mods are made, ensure that the active tab is the gallery tab
            new_collection = gallery_data is not None and ctx.triggered_id in ["data-collection", "uploaded_dict"]
            gallery_mod_in_tab = gallery_data is not None and ctx.triggered_id not in \
                          ["data-collection", "uploaded_dict", "annotation_canvas"] and \
                active_tab == 'gallery-tab'
            use_zoom = gallery_data is not None and ctx.triggered_id == 'annotation_canvas'
            if new_collection or gallery_mod_in_tab or use_zoom:
                row_children = []
                zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
                views = None
                # maintain the original order of channels that is dictated by the metadata
                # decide if channel view or ROI view is selected
                # channel view
                blend_return = dash.no_update
                if view_by_channel and channel_selected is not None:
                    views = get_all_images_by_channel_name(gallery_data, channel_selected)
                    if toggle_scaling_gallery:
                        try:
                            if blend_colour_dict[channel_selected]['x_lower_bound'] is None:
                                blend_colour_dict[channel_selected]['x_lower_bound'] = 0
                            if blend_colour_dict[channel_selected]['x_upper_bound'] is None:
                                blend_colour_dict[channel_selected]['x_upper_bound'] = \
                                get_default_channel_upper_bound_by_percentile(
                                    gallery_data[data_selection][channel_selected])
                            views = {key: apply_preset_to_array(resize_for_canvas(value),
                                                        blend_colour_dict[channel_selected]) for \
                                key, value in views.items()}
                        except KeyError:
                            pass
                else:
                    views = {elem: gallery_data[data_selection][elem] for elem in list(aliases.keys())}

                if views is not None:
                    for key, value in views.items():
                        if all([elem in canvas_layout for elem in zoom_keys]) and toggle_gallery_zoom:
                            x_range_low = math.floor(int(canvas_layout['xaxis.range[0]']))
                            x_range_high = math.floor(int(canvas_layout['xaxis.range[1]']))
                            y_range_low = math.floor(int(canvas_layout['yaxis.range[1]']))
                            y_range_high = math.floor(int(canvas_layout['yaxis.range[0]']))
                            assert x_range_high >= x_range_low
                            assert y_range_high >= y_range_low
                            try:
                                image_render = value[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                                                           range(int(x_range_low), int(x_range_high), 1))]
                            except IndexError as e:
                                image_render = value
                        else:
                            image_render = resize_for_canvas(value)
                        if toggle_scaling_gallery:
                            try:
                                if blend_colour_dict[key]['x_lower_bound'] is None:
                                    blend_colour_dict[key]['x_lower_bound'] = 0
                                if blend_colour_dict[key]['x_upper_bound'] is None:
                                    blend_colour_dict[key]['x_upper_bound'] = \
                                    get_default_channel_upper_bound_by_percentile(
                                    gallery_data[data_selection][key])
                                image_render = apply_preset_to_array(image_render,
                                                        blend_colour_dict[key])
                            except (KeyError, TypeError):
                                pass
                        if None not in (preset_selection, preset_dict) and nclicks > 0:
                            image_render = apply_preset_to_array(image_render, preset_dict[preset_selection])

                        label = aliases[key] if aliases is not None and key in aliases.keys() else key
                        row_children.append(dbc.Col(dbc.Card([dbc.CardBody(html.P(label, className="card-text")),
                                                          dbc.CardImg(
                                                              src=Image.fromarray(image_render).convert('RGB'),
                                                              bottom=True)]), width=3))
                return row_children, blend_return
            else:
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
        # current_blend = [elem['value'] for elem in current_blend] if current_blend is not None else None
        if current_blend is not None and len(current_blend) > 0:
            in_blend = [aliases[elem] for elem in current_blend]
            cell_styling_conditions = []
            if blend_colours is not None and current_blend is not None and data_selection is not None:
                for key in current_blend:
                    try:
                        if key in blend_colours.keys() and blend_colours[key]['color'] != '#FFFFFF':
                            label = aliases[key] if aliases is not None and key in aliases.keys() else key
                            cell_styling_conditions.append( {"condition": f"params.value == '{label}'",
                            "style": {"color": f"{blend_colours[key]['color']}"}})
                    except KeyError as e:
                        pass
                if len(in_blend) > 0:
                    to_return = pd.DataFrame(in_blend, columns=["Channel"]).to_dict(orient="records")
                    return to_return , {"sortable": False, "filter": False,
                         "cellStyle": {
                             "styleConditions": cell_styling_conditions}}
                else:
                    return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), \
                        {"sortable": False, "filter": False}
        else:
            return pd.DataFrame({}, columns=["Channel"]).to_dict(orient="records"), \
                        {"sortable": False, "filter": False}

    @dash_app.callback(
        Output("download-collapse", "is_open", allow_duplicate=True),
        [Input("open-download-collapse", "n_clicks")],
        [State("download-collapse", "is_open")])
    # @cache.memoize())
    def toggle_download_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

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
        else:
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
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis=XAxis(showticklabels=False),
                          yaxis=YAxis(showticklabels=False),
                          margin=dict(l=5, r=5, b=15, t=20, pad=0))
            cur_canvas['data'] = None
            return fig, go.Figure(cur_canvas), [None, None], [], None
        else:
            raise PreventUpdate

    @dash_app.callback(Output("pixel-hist", 'figure'),
                       Output('pixel-intensity-slider', 'max'),
                       Output('pixel-intensity-slider', 'value'),
                       Output('pixel-intensity-slider', 'marks'),
                       Output('blending_colours', 'data', allow_duplicate=True),
                       Input('images_in_blend', 'value'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       Input('blending_colours', 'data'),
                       Input("pixel-hist-collapse", "is_open"),
                       State('pixel-intensity-slider', 'value'),
                       prevent_initial_call=True)
                       # background=True,
                       # manager=cache_manager)
    # @cache.memoize())
    def create_pixel_histogram(selected_channel, uploaded, data_selection, current_blend_dict, show_pixel_hist,
                               cur_slider_values):
        """
        Create pixel histogram and output the default percentiles
        """
        if None not in (selected_channel, uploaded, data_selection, current_blend_dict):
            blend_return = dash.no_update
            try:
                if show_pixel_hist and ctx.triggered_id == "pixel-hist-collapse":
                    fig, hist_max = pixel_hist_from_array(uploaded[data_selection][selected_channel])
                    fig.update_layout(showlegend=False, yaxis={'title': None},
                                      xaxis={'title': None}, margin=dict(pad=0))
                else:
                    fig = dash.no_update
                    hist_max = int(np.max(uploaded[data_selection][selected_channel]))
            except (ValueError, TypeError) as e:
                fig = dash.no_update
                hist_max = 100
            spacing = int(hist_max / 3)
            tick_markers = dict([(round(i / 10) * 10, str(round(i / 10) * 10)) for i in range(0, hist_max, spacing)])
            # if the hist is triggered by the changing of a channel to modify or a new blend dict
            if ctx.triggered_id in ["images_in_blend"]:
                try:
                    # if the current selection has already had a histogram bound on it, update the histogram with it
                    if current_blend_dict[selected_channel]['x_lower_bound'] is not None and \
                            current_blend_dict[selected_channel]['x_upper_bound'] is not None:
                        lower_bound = int(float(current_blend_dict[selected_channel]['x_lower_bound']))
                        upper_bound = int(float(current_blend_dict[selected_channel]['x_upper_bound']))
                    else:
                        lower_bound = 0
                        upper_bound = get_default_channel_upper_bound_by_percentile(
                                        uploaded[data_selection][selected_channel])
                        current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                        current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                        blend_return = current_blend_dict
                    # set tick spacing between marks on the rangeslider
                    # have 4 tick markers
                    return fig, hist_max, [lower_bound, upper_bound], tick_markers, blend_return
                except (KeyError, ValueError) as e:
                    return {}, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif ctx.triggered_id == 'blending_colours':
                vals_return = dash.no_update
                if current_blend_dict[selected_channel]['x_lower_bound'] is not None and \
                        current_blend_dict[selected_channel]['x_upper_bound'] is not None:
                    if int(float(current_blend_dict[selected_channel]['x_lower_bound'])) != cur_slider_values[0] or \
                        int(float(current_blend_dict[selected_channel]['x_upper_bound'])) != cur_slider_values[1]:
                        lower_bound = int(float(current_blend_dict[selected_channel]['x_lower_bound']))
                        upper_bound = int(float(current_blend_dict[selected_channel]['x_upper_bound']))
                        vals_return = [lower_bound, upper_bound]
                else:
                    lower_bound = 0
                    upper_bound = get_default_channel_upper_bound_by_percentile(
                        uploaded[data_selection][selected_channel])
                    current_blend_dict[selected_channel]['x_lower_bound'] = lower_bound
                    current_blend_dict[selected_channel]['x_upper_bound'] = upper_bound
                    blend_return = current_blend_dict
                    vals_return = [lower_bound, upper_bound]
                return dash.no_update, hist_max, vals_return, tick_markers, blend_return
            elif ctx.triggered_id == "pixel-hist-collapse":
                return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            raise PreventUpdate

    @dash_app.callback(Output('bool-apply-filter', 'value'),
                       Output('filter-type', 'value'),
                       Output('kernel-val-filter', 'value'),
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
                       State("annotation-color-picker", 'value'))
    # @cache.memoize())
    def update_channel_filter_inputs(selected_channel, uploaded, data_selection, current_blend_dict,
                                     preset_selection, preset_dict, session_vars, cur_bool_filter, cur_filter_type,
                                     cur_filter_val, cur_colour):
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
            filter_type = current_blend_dict[selected_channel]['filter_type']
            filter_val = current_blend_dict[selected_channel]['filter_val']
            color = current_blend_dict[selected_channel]['color']
            # evaluate the current states of the inputs. if they are the same as the new channel, do not update
            if ' apply/refresh filter' in cur_bool_filter and None not in (filter_type, filter_val):
                to_apply_filter = dash.no_update
            else:
                to_apply_filter = [' apply/refresh filter'] if None not in (filter_type, filter_val) else []
            if filter_type == cur_filter_type:
                filter_type_return = dash.no_update
            else:
                filter_type_return = filter_type if filter_type is not None else "median"
            if filter_val == cur_filter_val:
                filter_val_return = dash.no_update
            else:
                filter_val_return = filter_val if filter_val is not None else 3
            if color == cur_colour['hex']:
                color_return = dash.no_update
            else:
                color_return = dict(hex=color) if color is not None and color not in \
                                                  ['#FFFFFF', '#ffffff'] else dash.no_update
            return to_apply_filter, filter_type_return, filter_val_return, color_return
        if ctx.triggered_id in ['preset-options'] and None not in \
                (preset_selection, preset_dict, selected_channel, data_selection, current_blend_dict):
            filter_type = preset_dict[preset_selection]['filter_type']
            filter_val = preset_dict[preset_selection]['filter_val']
            color = current_blend_dict[selected_channel]['color']
            to_apply_filter = [' apply/refresh filter'] if None not in (filter_type, filter_val) else []
            filter_type_return = filter_type if filter_type is not None else "median"
            filter_val_return = filter_val if filter_val is not None else 3
            color_return = dict(hex=color) if color is not None and color not in \
                                                  ['#FFFFFF', '#ffffff'] else dash.no_update
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
        else:
            raise PreventUpdate

    @dash_app.callback(Input('image_presets', 'data'),
                       Output('hover-preset-information', 'children'))
    # @cache.memoize())
    def update_hover_preset_information(preset_dict):
        """
        Update the hover information on the list of presets so that the user can preview the parameters before selecting
        """
        if preset_dict is not None and len(preset_dict) > 0:
            text = ''
            for stud, val in preset_dict.items():
                try:
                    low_bound = round(float(val['x_lower_bound']))
                except TypeError:
                    low_bound = None
                try:
                    up_bound = round(float(val['x_upper_bound']))
                except TypeError:
                    up_bound = None
                text = text + f"{stud}: \r\n l_bound: {low_bound}, " \
                              f"y_bound: {up_bound}, filter type: {val['filter_type']}, " \
                              f"filter val: {val['filter_val']} \r\n"

            return html.Textarea(text, style={"width": "200px", "height": f"{100 * len(preset_dict)}px"})
        else:
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
                assert all([elem in aliases.keys() for elem in session_config['unique_images']])
                return [{'label': aliases[i], 'value': i} for i in session_config['unique_images']]
            except (KeyError, AssertionError):
                return []
        else:
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
        else:
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
            if 'layout' in cur_layout.keys() and 'annotations' in cur_layout['layout'].keys() and \
                    len(cur_layout['layout']['annotations']) > 0:
                cur_layout['layout']['annotations'] = None
            if 'shapes' in cur_layout.keys():
                cur_layout['shapes'] = {}
            if 'layout' in cur_canvas.keys() and 'annotations' in cur_canvas['layout'].keys() and \
                    len(cur_canvas['layout']['annotations']) > 0:
                cur_canvas['layout']['annotations'] = None
            if 'layout' in cur_canvas.keys() and 'shapes' in cur_canvas['layout'].keys():
                cur_canvas['layout']['shapes'] = None

            fig = go.Figure(cur_canvas)
            fig.update_layout(dragmode='pan')
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                              xaxis=XAxis(showticklabels=False),
                              yaxis=YAxis(showticklabels=False),
                              margin=dict(
                                  l=0,
                                  r=0,
                                  b=0,
                                  t=0,
                                  pad=0
                              ))
            return fig, cur_layout
        else:
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
        else:
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
        else:
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
            if error_config is None:
                error_config = {"error": None}
            error_config["error"] = "Warning: the edited metadata appears to be incorrectly formatted. " \
                                    "Ensure that the number of " \
                                    "channels matches the provided channel labels."
            return error_config
        else:
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
        else:
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
        State('quant-annotation-col', 'value'))
    def add_annotation_to_dict(create_annotation, annotation_title, annotation_body, annotation_cell_type,
                               canvas_layout, annotations_dict, data_selection, cur_layers,
                               mask_toggle, mask_selection, mask_blending_level, add_mask_boundary, annot_col):
        if create_annotation and None not in (annotation_title, annotation_body,
                                              canvas_layout, data_selection, cur_layers):
            if annotations_dict is None or len(annotations_dict) < 1:
                annotations_dict = {}
            if data_selection not in annotations_dict.keys():
                annotations_dict[data_selection] = {}
            # use the data collection as the highest key for each ROI, then use the canvas coordinates to
            # uniquely identify a region

            # IMP: convert the dictionary to a sorted tuple to use as a key
            # https://stackoverflow.com/questions/1600591/using-a-python-dictionary-as-a-key-non-nested
            annotation_list = {}
            # Option 1: if zoom is used
            if isinstance(canvas_layout, dict) and 'shapes' not in canvas_layout:
                annotation_list[tuple(sorted(canvas_layout.items()))] = "zoom"
            # Option 2: if a shape is drawn on the canvas
            elif 'shapes' in canvas_layout and isinstance(canvas_layout, dict):
                # only get the shapes that are a rect or path, the others are canvas annotations
                for shape in canvas_layout['shapes']:
                    if shape['type'] == 'path':
                        annotation_list[shape['path']] = 'path'
                    elif shape['type'] == "rect":
                        key = {k: shape[k] for k in ('x0', 'x1', 'y0', 'y1')}
                        annotation_list[tuple(sorted(key.items()))] = "rect"
            for key, value in annotation_list.items():
                annotations_dict[data_selection][key] = {'title': annotation_title, 'body': annotation_body,
                                                               'cell_type': annotation_cell_type, 'imported': False,
                                                         'annotation_column': annot_col,
                                                            'type': value, 'channels': cur_layers,
                                                             'use_mask': mask_toggle,
                                                             'mask_selection': mask_selection,
                                                             'mask_blending_level': mask_blending_level,
                                                             'add_mask_boundary': add_mask_boundary}
            return Serverside(annotations_dict)
        else:
            raise PreventUpdate

    @dash_app.callback(Output('annotation-table', 'data'),
                       Output('annotation-table', 'columns'),
                       Input("annotations-dict", "data"),
                       Input('data-collection', 'value'),
                       prevent_initial_call=True)
    def populate_annotations_table_preview(annotations_dict, dataset_selection):
        if None not in (annotations_dict, dataset_selection):
            try:
                if len(annotations_dict[dataset_selection]) > 0:
                    annotation_list = []
                    for value in annotations_dict[dataset_selection].values():
                        for sub_key, sub_value in value.items():
                            value[sub_key] = str(sub_value)
                        annotation_list.append(value)
                    # columns = [{'id': p, 'name': p, 'editable': False} for p in annotations_dict[dataset_selection].keys()]
                    columns = [{'id': p, 'name': p, 'editable': False} for p in list(pd.DataFrame(annotation_list).columns)]
                    return annotation_list, columns
                else:
                    return [], []
            except KeyError:
                return [], []
        else:
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

    @dash_app.callback(
        Output("inputs-offcanvas", "is_open"),
        Input("inputs-offcanvas-button", "n_clicks"),
        State("inputs-offcanvas", "is_open"),
    )
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
    #     if ' show mask ID on hover' in show_mask_hover or \
    #             " show channel intensities on hover" in show_channel_intensity_hover:
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
        Input('add-annotation-col', 'n_clicks'),
        State('new-annotation-col', 'value'),
        State('quant-annotation-col', 'options'),
        prevent_initial_call=True)
    def add_new_annotation_column(nclicks, new_col, current_cols):
        """
        Add a new annotation column to the dropdown menu possibilities for quantification
        """
        if nclicks > 0 and new_col:
            try:
                assert isinstance(current_cols, list) and len(current_cols) > 0
                current_cols.append(new_col)
                return current_cols
            except AssertionError:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("annotations-dict", "data", allow_duplicate=True),
        Output('click-annotation-alert', 'children'),
        Output('click-annotation-alert', 'is_open'),
        Input('annotation_canvas', 'clickData'),
        State('click-annotation-assignment', 'value'),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        State('quant-annotation-col', 'value'),
        State('annotation_canvas', 'figure'),
        State('enable_click_annotation', 'value'),
        prevent_initial_call=True)
    def add_annotation_to_dict_with_click(clickdata, annotation_cell_type, annotations_dict,
                                          data_selection, annot_col, cur_figure, enable_click_annotation):

        if None not in (clickdata, data_selection, cur_figure) and enable_click_annotation and 'points' in clickdata:
            try:
                if annotations_dict is None or len(annotations_dict) < 1:
                    annotations_dict = {}
                if data_selection not in annotations_dict.keys():
                    annotations_dict[data_selection] = {}

                x = clickdata['points'][0]['x']
                y = clickdata['points'][0]['y']

                annotations_dict[data_selection][str(clickdata)] = {'title': None, 'body': None,
                                                 'cell_type': annotation_cell_type, 'imported': False,
                                                 'annotation_column': annot_col,
                                                 'type': "point", 'channels': None,
                                                 'use_mask': None,
                                                 'mask_selection': None,
                                                 'mask_blending_level': None,
                                                 'add_mask_boundary': None}
                return annotations_dict, html.H6(f"Point {x, y} updated with "
                                                 f"{annotation_cell_type} in {annot_col}"), True
            except KeyError:
                return dash.no_update, html.H6("Error in annotating point"), True
        else:
            raise PreventUpdate
