import os
import uuid
import dash
import pandas as pd
from dash import dcc
import matplotlib
from werkzeug.exceptions import BadRequest
import dash_uploader as du
from dash_extensions.enrich import Output, Input, State
from dash import ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from rakaia.inputs.pixel import set_roi_identifier_from_length
from rakaia.parsers.object import (
    RestyleDataParser,
    parse_and_validate_measurements_csv,
    parse_masks_from_filenames,
    parse_roi_query_indices_from_quantification_subset,
    get_quantification_filepaths_from_drag_and_drop,
    return_umap_dataframe_from_quantification_dict,
    read_in_mask_array_from_filepath,
    validate_imported_csv_annotations,
    GatingObjectList)
from rakaia.utils.decorator import DownloadDirGenerator
from rakaia.utils.object import (
    populate_quantification_frame_column_from_umap_subsetting,
    send_alert_on_incompatible_mask,
    ROIQuantificationMatch,
    validate_mask_shape_matches_image,
    quantification_distribution_table, custom_gating_id_list)
from rakaia.inputs.object import (
    generate_heatmap_from_interactive_subsetting,
    generate_umap_plot,
    umap_eligible_patch,
    patch_umap_figure,
    reset_custom_gate_slider)
from rakaia.io.pdf import AnnotationPDFWriter
from rakaia.io.annotation import AnnotationRegionWriter
from rakaia.utils.pixel import get_first_image_from_roi_dictionary
from rakaia.callbacks.object_wrappers import (
    AnnotationQuantificationMerge,
    callback_remove_canvas_annotation_shapes, reset_annotation_import)
from rakaia.io.annotation import AnnotationMaskWriter, export_point_annotations_as_csv
from rakaia.inputs.loaders import adjust_option_height_from_list_length
from rakaia.utils.pixel import split_string_at_pattern
from rakaia.io.readers import DashUploaderFileReader
from rakaia.utils.roi import generate_dict_of_roi_cell_ids
from rakaia.io.session import SessionServerside
from rakaia.utils.session import non_truthy_to_prevent_update
from rakaia.utils.cluster import (
    assign_colours_to_cluster_annotations,
    cluster_label_children,
    cluster_annotation_frame_import,
    set_cluster_col_dropdown,
    set_default_cluster_col,
    QuantificationClusterMerge,
    subset_cluster_frame)
from rakaia.utils.quantification import (
    populate_gating_dict_with_default_values,
    update_gating_dict_with_slider_values,
    gating_label_children,
    mask_object_counter_preview,
    DistributionTableColumns)

def init_object_level_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with object level analysis
    (object detection, quantification, dimensional reduction)

    :param dash_app: the dash proxy server wrapped in the parent Flask app
    :param tmpdirname: the path for the tmpdir for tmp storage for the session
    :param authentic_id: uuid string identifying the particular app invocation
    :param app_config: Dictionary of session options passed through CLI
    :return: None
    """
    dash_app.config.suppress_callback_exceptions = True
    matplotlib.use('agg')
    OVERWRITE = app_config['serverside_overwrite']
    dist_cols = [{'id': p, 'name': p, 'editable': False} for p in DistributionTableColumns.columns]

    @du.callback(Output('session_config_quantification', 'data'),
                 id='upload-quantification')
    def get_quantification_upload_from_drag_and_drop(status: du.UploadStatus):
        return get_quantification_filepaths_from_drag_and_drop(status)

    @dash_app.callback(Output('quantification-dict', 'data'),
                       Output('cell-type-col-designation', 'options'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('session_config_quantification', 'data'),
                       State('session_alert_config', 'data'),
                       prevent_initial_call=True)
    def populate_quantification_table_from_upload(session_dict, error_config):
        if session_dict is not None:
            quant_dict, cols, alert = parse_and_validate_measurements_csv(session_dict, error_config=error_config)
            return SessionServerside(quant_dict, key="quantification_dict", use_unique_key=OVERWRITE), cols, alert
        raise PreventUpdate

    @du.callback(Output('umap-projection', 'data'),
                 id='upload-umap-coordinates')
    def get_umap_upload_from_drag_and_drop(status: du.UploadStatus):
        files = DashUploaderFileReader(status).return_filenames()
        if files:
            return SessionServerside(pd.read_csv(files[0], names=['UMAP1', 'UMAP2'], header=0).to_dict(orient="records"),
                key="umap_coordinates", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @dash_app.callback(Output('quantification-heatmap-full', 'figure'),
                       Output('umap-legend-categories', 'data'),
                       Output('quantification-query-indices', 'data'),
                       Output('cur-umap-subset-category-counts', 'data'),
                       Output('query-cell-id-lists', 'data'),
                       Output('quant-heatmap-channel-list', 'options'),
                       Output('quant-heatmap-channel-list', 'value'),
                       Input('quantification-dict', 'data'),
                       Input('umap-plot', 'relayoutData'),
                       State('umap-projection', 'data'),
                       State('quant-annotation-col', 'options'),
                       Input('umap-plot', 'restyleData'),
                       Input('umap-projection-options', 'value'),
                       State('umap-legend-categories', 'data'),
                       Input('quant-heatmap-channel-list', 'value'),
                       State('quant-heatmap-channel-list', 'options'),
                       Input('normalize-heatmap', 'value'),
                       Input('subset-heatmap', 'value'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_heatmap(quantification_dict, umap_layout, embeddings, annot_cols, restyle_data,
                                            umap_col_selection, prev_categories, channels_to_display,
                                            heatmap_channel_options, normalize_heatmap, subset_heatmap):
        # figure out how to decouple the quantification update from the heatmap rendering:
        #  each time an annotation is added to the quant dictionary, the heatmap is re-rendered
        if quantification_dict is not None:
            zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
            if ctx.triggered_id not in ["umap-projection-options"]:
                try:
                    subtypes, keep = RestyleDataParser(restyle_data, quantification_dict,
                                    umap_col_selection, prev_categories).get_callback_structures()
                except TypeError:
                    subtypes, keep = None, None
            else:
                subtypes, keep = None, None
            try:
                fig, frame = generate_heatmap_from_interactive_subsetting(quantification_dict,
                        umap_layout, embeddings, zoom_keys, ctx.triggered_id, True, umap_col_selection,
                        subtypes, channels_to_display, normalize=normalize_heatmap, subset_val=subset_heatmap)
            except (BadRequest, IndexError):
                raise PreventUpdate
            indices_query, freq_counts_cat, cell_id_dict = None, None, None
            if frame is not None:
                indices_query, freq_counts_cat = parse_roi_query_indices_from_quantification_subset(
                    quantification_dict, frame, umap_col_selection)
                # also return the current count of the umap category selected to update the distribution table
                if umap_layout is not None:
                    merged = pd.DataFrame(quantification_dict).iloc[list(frame.index.values)]
                    cell_id_dict = generate_dict_of_roi_cell_ids(merged)
            # if the heatmap channel options are already set, do not update
            cols_selected = dash.no_update
            if ctx.triggered_id == "quantification-dict" and not heatmap_channel_options:
                cols_selected = list(frame.columns)
            return fig, keep, indices_query, freq_counts_cat, SessionServerside(cell_id_dict,
                    key="cell_id_list", use_unique_key=OVERWRITE), list(frame.columns), cols_selected
        raise PreventUpdate

    @dash_app.callback(Output('umap-projection', 'data', allow_duplicate=True),
                       Output('umap-projection-options', 'options'),
                       Output('gating-channel-options', 'options'),
                       Input('quantification-dict', 'data'),
                       State('umap-projection', 'data'),
                       Input('execute-umap-button', 'n_clicks'),
                       State('quant-heatmap-channel-list', 'value'),
                       prevent_initial_call=True)
    def generate_umap_from_measurements_csv(quantification_dict, current_umap, n_clicks, chan_include):
        """
        Generate a umap data frame projection of the measurements csv quantification. Returns a data frame
        of the embeddings and a list of the channels for interactive projection
        """
        if ctx.triggered_id == "quantification-dict":
            return dash.no_update, list(pd.DataFrame(quantification_dict).columns), list(pd.DataFrame(quantification_dict).columns)
        try:
            return return_umap_dataframe_from_quantification_dict(quantification_dict=quantification_dict, current_umap=
            current_umap, unique_key_serverside=OVERWRITE, cols_include=chan_include), dash.no_update, dash.no_update
        except ValueError: raise PreventUpdate

    @dash_app.callback(Output('umap-plot', 'figure'),
                       Output('umap-div-holder', 'style', allow_duplicate=True),
                       Input('umap-projection', 'data'),
                       Input('umap-projection-options', 'value'),
                       Input('quantification-dict', 'data'),
                       State('umap-plot', 'figure'),
                       Input('quantify-cur-roi-execute', 'n_clicks'),
                       prevent_initial_call=True)
    def plot_umap_for_measurements(embeddings, channel_overlay, quantification_dict, cur_umap_fig, trigger_quant):
        blank_umap = {'display': 'None'} if (len(pd.DataFrame(embeddings)) != len(pd.DataFrame(quantification_dict)) or
                                             ctx.triggered_id == 'quantify-cur-roi-execute') else dash.no_update
        if ctx.triggered_id != 'quantify-cur-roi-execute':
            if ctx.triggered_id == "umap-projection-options" and channel_overlay is None: return dash.no_update, blank_umap
            try:
                if umap_eligible_patch(cur_umap_fig, quantification_dict, channel_overlay):
                    return patch_umap_figure(quantification_dict, channel_overlay), {'display': 'inline-block'}
                else:
                    umap = generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig)
                    display = {'display': 'inline-block'} if isinstance(umap, go.Figure) else blank_umap
                return umap, display
            except BadRequest: return dash.no_update, blank_umap
        return dash.no_update, blank_umap

    @dash_app.callback(Output('quantification-dict', 'data', allow_duplicate=True),
                       Input('create-annotation-umap', 'n_clicks'),
                       State('quantification-dict', 'data'),
                       State('umap-projection', 'data'),
                       State('umap-plot', 'relayoutData'),
                       State('quant-annotation-col-in-tab', 'value'),
                       State('annotation-cell-types-quantification', 'value'),
                       prevent_initial_call=True)
    def add_annotation_column_using_umap_subsetting(add_annotation, measurements, embeddings,
                                                    umap_layout, annot_col, annot_value):
        """
        Add an annotation column to the quantification frame using interactive UMAP subsetting. The annotation will
        be applied to the current cells in the UMAP frame
        """
        if None not in (measurements, annot_col, annot_value, umap_layout) and add_annotation:
            return SessionServerside(populate_quantification_frame_column_from_umap_subsetting(
                pd.DataFrame(measurements), pd.DataFrame(embeddings), umap_layout, annot_col, annot_value).to_dict(
                orient='records'), key="quantification_dict", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @du.callback(Output('mask-uploads', 'data'),
                 id='upload-mask')
    def return_mask_upload(status: du.UploadStatus):
        return parse_masks_from_filenames(status)

    @dash_app.callback(Output('input-mask-name', 'value'),
                       Input('mask-uploads', 'data'),
                       Input('mask-name-autofill', 'n_clicks'),
                       State('data-collection', 'value'),
                       State('dataset-delimiter', 'value'),
                       prevent_initial_call=True)
    def set_import_single_mask_upload(mask_uploads, autofill_mask_name, data_selection, delimiter):
        """
        Allow the user to change the mask upload name if only a single mask is uploaded
        If multiple are uploaded, use the file basename by default
        Inputs are either triggered by the upload with the basename, or the autofill using the current ROI identifier
        """
        if ctx.triggered_id == 'mask-uploads' and mask_uploads is not None and len(mask_uploads) == 1:
            return list(mask_uploads.keys())[0]
        elif ctx.triggered_id == 'mask-name-autofill' and data_selection and autofill_mask_name:
            return str(set_roi_identifier_from_length(data_selection, delimiter=delimiter))
        raise PreventUpdate

    @dash_app.callback(Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('mask-options', 'value', allow_duplicate=True),
                       Input('mask-dict', 'data'),
                       State('data-collection', 'value'),
                       Input('uploaded_dict', 'data'),
                       State('session_alert_config', 'data'),
                       Input('mask-options', 'value'),
                       Input('apply-mask', 'value'),
                       prevent_initial_call=True)
    def give_alert_on_improper_mask_import(mask_dict, data_selection, upload_dict, error_config, mask_selection,
                                           mask_toggle):
        """
        Send an alert if the imported mask does not match the current ROI selection
        Works by validating the imported mask against the first channel of the current ROI selection
        """
        return send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection, mask_toggle)

    @dash_app.callback(
        Output("mask-name-modal", "is_open"),
        Input('input-mask-name', 'value'),
        Input('set-mask-name', 'n_clicks'),
        State('mask-uploads', 'data'))
    def toggle_mask_name_input_modal(new_mask_name, mask_clicks, mask_uploads):
        # only show the modal if the mask uploads len is 1
        if new_mask_name and ctx.triggered_id == "input-mask-name" and len(mask_uploads) == 1: return True
        return False

    @dash_app.callback(Output('mask-dict', 'data'),
                       Output('mask-options', 'options'),
                       Output('mask-options', 'optionHeight'),
                       Input('mask-uploads', 'data'),
                       State('input-mask-name', 'value'),
                       Input('set-mask-name', 'n_clicks'),
                       State('mask-dict', 'data'),
                       prevent_initial_call=True)
    def set_mask_dict_and_options(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict):
        # cases where the callback should occur: if the mask dict is longer than 1 and triggered by the dictionary
        # or, if there is a single mask and the trigger is setting the mask name
        multi_upload = ctx.triggered_id == "mask-uploads" and len(mask_uploads) > 1
        single_upload = ctx.triggered_id == 'set-mask-name' and len(mask_uploads) == 1
        if multi_upload or single_upload:
            mask_dict, options = read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name, set_mask,
                                                                  cur_mask_dict, unique_key_serverside=OVERWRITE)
            # if any of the names are longer than 40 characters, increase the height to make them visible
            height_update = adjust_option_height_from_list_length(options, dropdown_type="mask")
            return mask_dict, options, height_update
        raise PreventUpdate

    @dash_app.callback(
        Input("annotations-dict", "data"),
        State('quantification-dict', 'data'),
        State('data-collection', 'value'),
        State('data-collection', 'options'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        State('dataset-delimiter', 'value'),
        Input('delete-annotation-tabular', 'n_clicks'),
        State('annotation-table', 'selected_rows'),
        Input('quant-annot-reimport', 'n_clicks'),
        Input('clear-annotation_dict', 'n_clicks'),
        Output('quantification-dict', 'data', allow_duplicate=True),
        Output("annotations-dict", "data", allow_duplicate=True),
        Output('annotation-table', 'selected_rows', allow_duplicate=True))
    def update_region_annotation_in_quantification_frame(annotations, quantification_frame,
                        data_selection, data_dropdown_options, mask_config, mask_toggle, mask_selection, delimiter,
                        delete_from_table, annot_table_selection, reimport_annots, clear_all_annots):
        """
        Add or remove region annotation to the segmented objects of a quantification data frame
        Undoing an annotation both removes it from the annotation hash, and the quantification frame if it exists
        Any selected rows in the annotation preview table are reset to avoid erroneous indices
        """
        if data_selection:
            remove = ctx.triggered_id in ["delete-annotation-tabular", "clear-annotation_dict"]
            if ctx.triggered_id == "clear-annotation_dict" and data_selection in annotations:
                indices_remove = [int(i) for i in range(len(annotations[data_selection].keys()))]
            else:
                indices_remove = annot_table_selection if ctx.triggered_id == "delete-annotation-tabular" else None
            sample_name, id_column = ROIQuantificationMatch(data_selection, quantification_frame,
                        data_dropdown_options, delimiter, mask_selection).get_matches()
            if ctx.triggered_id == "quant-annot-reimport" and reimport_annots:
                annotations = reset_annotation_import(annotations, data_selection, app_config, False)
            quant_frame, annotations = AnnotationQuantificationMerge(annotations, quantification_frame, data_selection,
            mask_config, mask_toggle, mask_selection, sample_name=sample_name, id_column=id_column,
            config=app_config, remove=remove, indices_remove=indices_remove).get_callback_structures()
            return SessionServerside(quant_frame, key="quantification_dict", use_unique_key=OVERWRITE), annotations, []
        raise PreventUpdate

    @dash_app.callback(
        Output("download-edited-annotations", "data"),
        Input("btn-download-annotations", "n_clicks"),
        Input("quantification-dict", "data"))
    def download_quantification_with_annotations(n_clicks, datatable_contents):
        if n_clicks is not None and n_clicks > 0 and datatable_contents is not None and \
                ctx.triggered_id == "btn-download-annotations":
            return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "measurements.csv", index=False)
        raise PreventUpdate

    @dash_app.callback(
        Output("download-annotation-pdf", "data"),
        Input("btn-download-annot-pdf", "n_clicks"),
        State("annotations-dict", "data"),
        State('canvas-layers', 'data'),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('alias-dict', 'data'),
        State('blending_colours', 'data'),
        State('bool-apply-global-filter', 'value'),
        State('global-filter-type', 'value'),
        State("global-kernel-val-filter", 'value'),
        State("global-sigma-val-filter", 'value'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def download_annotations_pdf(download_pdf, annotations_dict, canvas_layers, data_selection, mask_config, aliases,
                blend_dict, global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma):
        if download_pdf and None not in (annotations_dict, canvas_layers, data_selection):
            return dcc.send_file(non_truthy_to_prevent_update(AnnotationPDFWriter(download_pdf, annotations_dict, canvas_layers,
                data_selection, mask_config, aliases, "annotations.pdf", blend_dict, global_apply_filter,
                global_filter_type, global_filter_val, global_filter_sigma).generate_annotation_pdf()), type="application/pdf")
        raise PreventUpdate

    @dash_app.callback(
        Output("download-annotation-mask", "data"),
        Input("btn-download-annot-mask", "n_clicks"),
        State("annotations-dict", "data"),
        State('canvas-layers', 'data'),
        State('data-collection', 'value'),
        State('uploaded_dict', 'data'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads', 'annotation_masks'))
    def download_annotations_masks(download_mask, annotations_dict, canvas_layers,
                                 data_selection, image_dict, mask_dict, apply_mask, mask_selection):
        if download_mask and None not in (annotations_dict, canvas_layers, data_selection, image_dict) and \
                data_selection in annotations_dict and len(annotations_dict[data_selection]) > 0:
            first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
            # check that the mask is compatible with the current image
            if None not in (mask_dict, mask_selection) and apply_mask and validate_mask_shape_matches_image(first_image,
                                                                                mask_dict[mask_selection]['raw']):
                mask_used = mask_dict[mask_selection]['raw']
            else:
                mask_used = None
            return dcc.send_file(AnnotationMaskWriter(download_mask, annotations_dict, data_selection,
                (first_image.shape[0], first_image.shape[1]), mask_used, False).write_annotation_masks())
        raise PreventUpdate

    @dash_app.callback(
        Output('annotation_canvas', 'figure', allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Input("clear-region-annotation-shapes", "n_clicks"),
        State('annotation_canvas', 'figure'),
        State('annotation_canvas', 'relayoutData'),
        State('session_alert_config', 'data'))
    def clear_canvas_shapes(n_clicks, cur_canvas, canvas_layout, error_config):
        """
        Clear the current canvas of any shapes that are not associated with the legend or scalebar
        Important: requires that the current drag mode be set to zoom or pan to remove any shapes in the current
        canvas layout
        """
        return callback_remove_canvas_annotation_shapes(n_clicks, cur_canvas, canvas_layout, error_config)

    @dash_app.callback(
        Output("show-quant-dist-table", "is_open"),
        Input('show-quant-dist', 'n_clicks'),
        [State("show-quant-dist-table", "is_open")])
    def toggle_show_quant_dist_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(Output('quant-dist-table', 'data'),
                       Output('quant-dist-table', 'columns'),
                       Input("umap-projection-options", "value"),
                       Input('quantification-dict', 'data'),
                       Input('cur-umap-subset-category-counts', 'data'),
                       prevent_initial_call=True)
    def populate_quantification_distribution_table(umap_variable, quantification_dict, subset_cur_cat):
        if None not in (quantification_dict, umap_variable):
            return quantification_distribution_table(quantification_dict, umap_variable, subset_cur_cat), dist_cols
        return pd.DataFrame({}).to_dict(orient="records"), dist_cols

    @dash_app.callback(
        Output("download-point-csv", "data"),
        Input("btn-download-points-csv", "n_clicks"),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        State('uploaded_dict', 'data'),
        State('dataset-delimiter', 'value'),
        prevent_initial_call=True)
    def download_point_annotations_as_csv(n_clicks, annotations_dict, data_selection,
                                          mask_dict, apply_mask, mask_selection, image_dict, delimiter):
        if data_selection and annotations_dict and image_dict and delimiter:
            exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
            return export_point_annotations_as_csv(n_clicks, acq, annotations_dict, data_selection, mask_dict, apply_mask,
                                        mask_selection, image_dict, authentic_id, tmpdirname, delimiter, True)
        raise PreventUpdate

    @dash_app.callback(
        Output("download-region-csv", "data"),
        Input("btn-download-region-csv", "n_clicks"),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('dataset-delimiter', 'value'),
        prevent_initial_call=True)
    def download_region_annotations_as_csv(n_clicks, annotations_dict, data_selection, mask_dict, delimiter):
        if n_clicks and None not in (annotations_dict, data_selection):
            download_dir = os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads')
            return dcc.send_file(non_truthy_to_prevent_update(AnnotationRegionWriter(
                annotations_dict, data_selection, mask_dict, delimiter).write_csv(dest_dir=download_dir)))
        raise PreventUpdate


    @dash_app.callback(
        Output("download-umap-projection", "data"),
        Input("btn-download-umap-projection", "n_clicks"),
        State('umap-projection', 'data'),
        prevent_initial_call=True)
    def download_umap_projection_csv(execute_download, umap_projection):
        """
        Download the current UMAP coordinates (UMAP1 and 2) in CSV format
        """
        if execute_download > 0 and umap_projection is not None:
            return dcc.send_data_frame(pd.DataFrame(umap_projection, columns=['UMAP1', 'UMAP2']).to_csv,
                                       "umap_coordinates.csv", index=False)
        raise PreventUpdate

    @dash_app.callback(
        Output("umap-config-modal", "is_open"),
        Input('umap-config-button', 'n_clicks'),
        Input('execute-umap-button', 'n_clicks'),
        [State("umap-config-modal", "is_open")])
    def toggle_show_umap_config_modal(n, render, is_open):
        return not is_open if (n or render) else is_open

    @dash_app.callback(
        Output("heatmap-config-modal", "is_open"),
        Input('heatmap-config-button', 'n_clicks'),
        [State("heatmap-config-modal", "is_open")])
    def toggle_show_heatmap_config_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(
        Output("quantification-roi-modal", "is_open"),
        Output('mask-object-counter', 'children'),
        Input('quantify-cur-roi-button', 'n_clicks'),
        Input('quantify-cur-roi-execute', 'n_clicks'),
        State("quantification-roi-modal", "is_open"),
        State('channel-quantification-list', 'value'),
        State('mask-dict', 'data'),
        Input('mask-options', 'value'))
    def toggle_show_quantification_config_modal(n, execute, is_open, channels_to_quantify, mask_dict, mask_selection):
        preview = mask_object_counter_preview(mask_dict, mask_selection)
        if ctx.triggered_id == "mask-options": return dash.no_update, preview
        if ctx.triggered_id == "quantify-cur-roi-execute" and execute > 0 and channels_to_quantify: return False, preview
        return not is_open if n else is_open, preview

    @du.callback(Output('imported-annotations-csv', 'data'),
                 id='upload-point-annotations')
    def import_point_annotations_from_drag_and_drop(status: du.UploadStatus):
        """
        Import a CSV of point annotations to re-render on the canvas
        """
        files = DashUploaderFileReader(status).return_filenames()
        if files:
            frame = pd.read_csv(files[0])
            if validate_imported_csv_annotations(frame):
                return SessionServerside(frame.to_dict(orient="records"), key="point_annotations", use_unique_key=OVERWRITE)
            raise PreventUpdate
        raise PreventUpdate

    @du.callback(Output('uploads_cluster', 'data', allow_duplicate=True),
                 id='upload-cluster-annotations')
    def get_filenames_from_cluster_drag_and_drop(status: du.UploadStatus):
        files = DashUploaderFileReader(status).return_filenames()
        return files if files else dash.no_update

    @dash_app.callback(
                 Output('imported-cluster-frame', 'data', allow_duplicate=True),
                 Output('cluster-col', 'options'),
                 Input('uploads_cluster', 'data'),
                 State('imported-cluster-frame', 'data'),
                 State('data-collection', 'value'))
    def get_cluster_assignment_upload_from_drag_and_drop(uploads, cur_clusters, data_selection):
        """
        Parse a frame of cluster mask object projections in CSV format
        """
        if uploads and data_selection:
            return SessionServerside(cluster_annotation_frame_import(cur_clusters, data_selection,
            pd.read_csv(uploads[0])), key="cluster_assignments", use_unique_key=OVERWRITE), \
                set_cluster_col_dropdown(pd.read_csv(uploads[0]))
        raise PreventUpdate

    @dash_app.callback(
                Output('cluster-colour-assignments-dict', 'data'),
                Input('imported-cluster-frame', 'data'),
                State('data-collection', 'value'),
                State('cluster-colour-assignments-dict', 'data'))
    def assign_cluster_colors(cluster_frame, data_selection, cur_cluster_dict):
        """
        Assign colors to cluster values from a selected cluster category. Will auto-assign all the categories
        provided in the frame
        """
        if cluster_frame and data_selection and data_selection in cluster_frame:
            return assign_colours_to_cluster_annotations(cluster_frame, cur_cluster_dict, data_selection)
        raise PreventUpdate

    @dash_app.callback(
        Output('cluster-label-list', 'options'),
        Output('cluster-label-selection', 'options'),
        Output('cluster-label-selection', 'value'),
        Output('cluster-assignments', 'children'),
        Input('cluster-col', 'value'),
        State('data-collection', 'value'),
        Input('cluster-colour-assignments-dict', 'data'))
    def generate_clust_selection_options_legend(clust_select, data_selection, cur_cluster_dict):
        """
        Generate the cluster color label selection on a category selection
        """
        if clust_select and data_selection and clust_select in cur_cluster_dict[data_selection]:
            options = list(cur_cluster_dict[data_selection][clust_select].keys())
            return options, options, options, cluster_label_children(data_selection, cur_cluster_dict, clust_select)
        return [], [], [], []

    @dash_app.callback(
        Output('cluster-label-selection', 'value', allow_duplicate=True),
        Input('toggle-clust-selection', 'value'),
        State('cluster-label-selection', 'options'))
    def generate_cluster_colour_assignment(toggle_clust_selection, clust_options):
        """
            Toggle the options selectable for a specific cluster category
            """
        return clust_options if toggle_clust_selection else []

    @dash_app.callback(Input('data-collection', 'value'),
                       State('cluster-colour-assignments-dict', 'data'),
                       State('imported-cluster-frame', 'data'),
                       Output('cluster-col', 'options', allow_duplicate=True),
                       Output('cluster-col', 'value', allow_duplicate=True))
    def update_cluster_assignment_options_on_data_selection_change(data_selection, cluster_frame, master_clust):
        """
        Update the cluster categories selectable on an ROI change
        """
        if cluster_frame and data_selection and data_selection in cluster_frame and master_clust and \
                data_selection in master_clust:
            return list(cluster_frame[data_selection].keys()), set_default_cluster_col(cluster_frame, data_selection)
        return [], None

    @dash_app.callback(Output('cluster-colour-assignments-dict', 'data', allow_duplicate=True),
                       Input('cluster-color-picker', 'value'),
                       State('cluster-label-list', 'value'),
                       State('data-collection', 'value'),
                       State('cluster-colour-assignments-dict', 'data'),
                       State('cluster-col', 'value'))
    def assign_colour_to_cluster_label(colour_selection, cluster_selection, data_selection, clust_dict, cluster_cat):
        """
        Assign the designated colour from the colour picker to the selected cluster label
        """
        if colour_selection and data_selection and clust_dict and cluster_cat and cluster_selection is not None:
            try:
                clust_dict[data_selection][cluster_cat][cluster_selection] = colour_selection['hex']
                return clust_dict
            except KeyError: raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output('cluster-label-collapse', 'is_open'),
        [Input('toggle-cluster-labels', 'n_clicks')],
        [State('cluster-label-collapse', 'is_open')])
    def toggle_pixel_hist_collapse(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(Output('gating-cur-mod', 'options'),
                       Output('gating-dict', 'data'),
                       Output('gating-cur-mod', 'value'),
                       Input('gating-channel-options', 'value'),
                       State('gating-dict', 'data'),
                       Input('quantification-dict', 'data'))
    def update_current_gating_selection(gating_selection, current_gate_dict, quant_dict):
        """
        Update the dropdown options for the current gating selection. The multi-selection will define
        the current set of parameters on which the mask is gated
        """
        if gating_selection:
            gates_to_keep = [gate for gate in gating_selection if gate in pd.DataFrame(quant_dict).columns]
            return gates_to_keep, populate_gating_dict_with_default_values(current_gate_dict, gating_selection), \
                gates_to_keep[-1] if len(gates_to_keep) > 0 else None
        return dash.no_update, dash.no_update, None

    @dash_app.callback(Output('gating-slider', 'value'),
                       Input('gating-cur-mod', 'value'),
                       State('gating-dict', 'data'),
                       State('quantification-dict', 'data'))
    def update_gating_thresholds(gate_selected, gating_dict, quantification_dict):
        """
        Update the values shown in the gating range slider when a parameter is selected for mod
        """
        if None not in (gate_selected, quantification_dict):
            if gating_dict and gate_selected in gating_dict:
                return gating_dict[gate_selected]['lower_bound'], gating_dict[gate_selected]['upper_bound']
            return [0.0, 1.0]
        raise PreventUpdate


    @dash_app.callback(Output('gating-dict', 'data', allow_duplicate=True),
                       Input('gating-slider', 'value'),
                       State('gating-dict', 'data'),
                       State('gating-cur-mod', 'value'))
    def update_gating_dict(gating_val, gating_dict, gate_selected):
        """
        update the gating dictionary when a parameter has its values changed
        """
        if None not in (gating_val, gate_selected):
            gating_dict = update_gating_dict_with_slider_values(gating_dict, gate_selected, gating_val)
            return SessionServerside(gating_dict, key="gating_dict", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @dash_app.callback(Output('gating-cell-list', 'data'),
                       Output('gating-param-display', 'children'),
                       Output('apply-gating-custom', 'value'),
                       Input('gating-dict', 'data'),
                       Input('data-collection', 'value'),
                       State('quantification-dict', 'data'),
                       Input('mask-options', 'value'),
                       Input('gating-channel-options', 'value'),
                       Input('gating-blend-type', 'value'),
                       Input('custom-id-gating', 'value'),
                       Input('apply-gating-custom', 'value'))
    def update_gating_object_list(gating_dict, roi_selection, quantification_dict, mask_selection,
                                cur_gate_selection, gating_type, id_str, apply_custom_gating):
        # do not update if using custom list and the parameters are updated
        if ctx.triggered_id in ["gating-dict"] and apply_custom_gating: raise PreventUpdate
        elif ctx.triggered_id in ['custom-id-gating', 'apply-gating-custom'] and id_str and apply_custom_gating:
            id_list = custom_gating_id_list(id_str)
            return SessionServerside(id_list, key="gating_cell_id_list", use_unique_key=OVERWRITE), \
                gating_label_children(False, None, None, id_list, True), dash.no_update
        elif None not in (roi_selection, quantification_dict, mask_selection) and cur_gate_selection:
            id_list = GatingObjectList(gating_dict, cur_gate_selection, pd.DataFrame(quantification_dict),
                        mask_selection, intersection=(gating_type == 'intersection')).get_object_list()
            return SessionServerside(id_list, key="gating_cell_id_list", use_unique_key=OVERWRITE), \
                gating_label_children(True, gating_dict, cur_gate_selection, id_list), reset_custom_gate_slider(ctx.triggered_id)
        return [] if gating_dict is not None else dash.no_update, [], dash.no_update if cur_gate_selection else False

    @dash_app.callback(Output('imported-cluster-frame', 'data', allow_duplicate=True),
                       Output('cluster-col', 'options', allow_duplicate=True),
                       Input('overlay-to-clust', 'n_clicks'),
                       State('data-collection', 'value'),
                       State('quantification-dict', 'data'),
                       State('dataset-delimiter', 'value'),
                       State('umap-projection-options', 'value'),
                       State('imported-cluster-frame', 'data'),
                       State('mask-options', 'value'))
    def transfer_quant_to_clust(transfer, roi_selection, quant_dict, delimiter, overlay, cur_frame, cur_mask):
        if transfer and roi_selection and quant_dict and delimiter and overlay:
            clust = QuantificationClusterMerge(quant_dict, roi_selection, overlay, cur_frame, delimiter, cur_mask).get_cluster_frame()
            return SessionServerside(clust, key="cluster_assignments", use_unique_key=OVERWRITE), set_cluster_col_dropdown(clust[roi_selection])
        raise PreventUpdate

    @dash_app.callback(
        Output("show-clust-dist-table", "is_open"),
        Input('toggle-cluster-dist', 'n_clicks'),
        [State("show-clust-dist-table", "is_open")])
    def toggle_show_clust_dist_modal(n, is_open):
        return not is_open if n else is_open

    @dash_app.callback(Output('clust-dist-table', 'data'),
                       Output('clust-dist-table', 'columns'),
                       State('cluster-col', 'value'),
                       State('imported-cluster-frame', 'data'),
                       State('data-collection', 'value'),
                       Input('cluster-label-selection', 'value'),
                       Input('gating-cell-list', 'data'),
                       Input('apply-gating', 'value'),
                       prevent_initial_call=True)
    def populate_cluster_projection_distribution_table(clust_variable, cluster_data, roi_selection, cluster_cats,
                                                       gating_object_list, use_gating):
        if cluster_data and clust_variable and cluster_cats and roi_selection:
            gating_object_list = gating_object_list if use_gating else None
            cluster_data = subset_cluster_frame(cluster_data, roi_selection, clust_variable, cluster_cats, gating_object_list)
            return quantification_distribution_table(cluster_data, clust_variable, None), dist_cols
        return pd.DataFrame({}).to_dict(orient="records"), dist_cols
