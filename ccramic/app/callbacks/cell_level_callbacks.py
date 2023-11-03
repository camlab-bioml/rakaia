import dash
import pandas as pd

from .cell_level_wrappers import *
from ..io.annotation_outputs import *
from dash import dcc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from werkzeug.exceptions import BadRequest


def init_cell_level_callbacks(dash_app, tmpdirname, authentic_id):
    """
    Initialize the callbacks associated with cell level analysis (object detection, quantification, dimensional reduction)
    """
    dash_app.config.suppress_callback_exceptions = True
    matplotlib.use('agg')

    @du.callback(Output('session_config_quantification', 'data'),
                 id='upload-quantification')
    # @cache.memoize())
    def get_quantification_upload_from_drag_and_drop(status: du.UploadStatus):
        return get_quantification_filepaths_from_drag_and_drop(status)

    @dash_app.callback(Output('quantification-dict', 'data'),
                       Output('cell-type-col-designation', 'options'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Input('session_config_quantification', 'data'),
                       State('session_alert_config', 'data'),
                       State('uploaded_dict', 'data'),
                       State('data-collection', 'value'),
                       prevent_initial_call=True)
    def populate_quantification_table_from_upload(session_dict, error_config, upload_dict, data_selection):
        if None not in (data_selection, upload_dict):
            first_image = list(upload_dict[data_selection].keys())[0]
            image_for_validation = upload_dict[data_selection][first_image]
        else:
            image_for_validation = None
        quant_dict, cols, alert = parse_and_validate_measurements_csv(session_dict, error_config=error_config,
                                                   image_to_validate=image_for_validation)
        return Serverside(quant_dict), cols, alert

    @du.callback(Output('umap-projection', 'data'),
                 id='upload-umap-coordinates')
    # @cache.memoize())
    def get_quantification_upload_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        if filenames and float(status.progress) == 1.0:
            frame = pd.read_csv(filenames[0])
            if len(frame.columns) == 2:
                frame.columns = ['UMAP1', 'UMAP2']
                return Serverside(frame.to_dict(orient="records"))
            else:
                raise PreventUpdate

    @dash_app.callback(Output('quantification-bar-full', 'figure'),
                       Output('umap-legend-categories', 'data'),
                       Output('quantification-query-indices', 'data'),
                       Output('cur-umap-subset-category-counts', 'data'),
                       Input('quantification-dict', 'data'),
                       State('annotation_canvas', 'relayoutData'),
                       Input('quantification-bar-mode', 'value'),
                       Input('umap-plot', 'relayoutData'),
                       State('umap-projection', 'data'),
                       State('quant-annotation-col', 'options'),
                       Input('umap-plot', 'restyleData'),
                       Input('umap-projection-options', 'value'),
                       State('umap-legend-categories', 'data'),
                       State('dynamic-update-barplot', 'value'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_statistics(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, annot_cols, restyle_data, umap_col_selection,
                                               prev_categories, dynamic_update):
        #TODO: incorporate subsetting based on legend selection
        # uses the restyledata for the current legend selection to figure out which selections have been made
        # Example 1: user selected only the third legend item to view
        # [{'visible': ['legendonly', 'legendonly', True, 'legendonly', 'legendonly', 'legendonly', 'legendonly']}, [0, 1, 2, 3, 4, 5, 6]]
        # Example 2: user selects all but the the second item to view
        # [{'visible': ['legendonly']}, [2]]
        # print(restyle_data)

        # TODO: fix when umap layout resets after tab switch
        # do not update if the tab is switched and the umap layout is reset to the default
        # tab_switch = ctx.triggered_id == "umap-plot" and umap_layout in [{"autosize": True}]
        if quantification_dict is not None:
            zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]','yaxis.range[0]', 'yaxis.range[1]']
            if ctx.triggered_id not in ["umap-projection-options"]:
                try:
                    subtypes, keep = parse_cell_subtypes_from_restyledata(restyle_data, quantification_dict, umap_col_selection,
                                                              prev_categories)
                except TypeError:
                    subtypes, keep = None, None
            else:
                subtypes, keep = None, None

            try:
                # do not update the expression barplot if the feature is turned off
                if not (ctx.triggered_id == "umap-layout" and len(dynamic_update) <= 0):
                    fig, frame = generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, zoom_keys, ctx.triggered_id, annot_cols,
                                                                        umap_col_selection, subtypes)
                else:
                    fig, frame = dash.no_update, dash.no_update
            except (BadRequest, IndexError):
                raise PreventUpdate
            if frame is not None:
                indices_query, freq_counts_cat = parse_roi_query_indices_from_quantification_subset(
                    quantification_dict, frame, umap_col_selection)
                    # also return the current count of the uamp category selected to update the distribution table
            else:
                indices_query = None
                freq_counts_cat = None
            return fig, keep, indices_query, freq_counts_cat
        else:
            raise PreventUpdate


    @dash_app.callback(Output('umap-projection', 'data', allow_duplicate=True),
                       Output('umap-projection-options', 'options'),
                       Input('quantification-dict', 'data'),
                       State('umap-projection', 'data'),
                       Input('execute-umap-button', 'n_clicks'),
                       prevent_initial_call=True)
    def generate_umap_from_measurements_csv(quantification_dict, current_umap, n_clicks):
        """
        Generate a umap data frame projection of the measurements csv quantification. Returns a data frame
        of the embeddings and a list of the channels for interactive projection
        """
        if ctx.triggered_id == "quantification-dict":
            return dash.no_update, list(pd.DataFrame(quantification_dict).columns)
        else:
            try:
                if n_clicks > 0:
                    return return_umap_dataframe_from_quantification_dict(quantification_dict=quantification_dict,
                                                                  current_umap=current_umap)
                else:
                    raise PreventUpdate
            except ValueError:
                return dash.no_update, list(pd.DataFrame(quantification_dict).columns)

    @dash_app.callback(Output('umap-plot', 'figure'),
                       Input('umap-projection', 'data'),
                       Input('umap-projection-options', 'value'),
                       State('quantification-dict', 'data'),
                       State('umap-plot', 'figure'),
                       prevent_initial_call=True)
    def plot_umap_for_measurements(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
        try:
            return generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig)
        except BadRequest:
            raise PreventUpdate

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
        Add an annotation column to the quantification frame using interactive UMAp subsetting. The annotation will
        be applied to the current cells in the UMAP frame
        """
        if None not in (measurements, annot_col, annot_value, umap_layout) and add_annotation > 0:
            return Serverside(populate_quantification_frame_column_from_umap_subsetting(
                            pd.DataFrame(measurements), pd.DataFrame(embeddings), umap_layout, annot_col,
                annot_value).to_dict(orient='records'))
        else:
            raise PreventUpdate



    @du.callback(Output('mask-uploads', 'data'),
                 id='upload-mask')
    # @cache.memoize())
    def return_mask_upload(status: du.UploadStatus):
        return parse_masks_from_filenames(status)

    @dash_app.callback(Output('input-mask-name', 'value'),
                       Input('mask-uploads', 'data'),
                       prevent_initial_call=True)
    def input_mask_name_on_upload(mask_uploads):
        if mask_uploads is not None and len(mask_uploads) > 0 and len(mask_uploads) == 1:
            return list(mask_uploads.keys())[0]
        else:
            raise PreventUpdate

    @dash_app.callback(Output('session_alert_config', 'data', allow_duplicate=True),
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
        return send_alert_on_incompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection,
                                           mask_toggle)

    @dash_app.callback(
        Output("mask-name-modal", "is_open"),
        Input('input-mask-name', 'value'),
        Input('set-mask-name', 'n_clicks'),
        State('mask-uploads', 'data'))
    def toggle_mask_name_input_modal(new_mask_name, mask_clicks, mask_uploads):
        # only show the modal if the mask uploads len is 1
        if new_mask_name and ctx.triggered_id == "input-mask-name" and len(mask_uploads) == 1:
            return True
        elif ctx.triggered_id == "set-mask-name" and mask_clicks > 0:
            return False
        else:
            return False

    @dash_app.callback(Output('mask-dict', 'data'),
                       Output('mask-options', 'options'),
                       Input('mask-uploads', 'data'),
                       State('input-mask-name', 'value'),
                       Input('set-mask-name', 'n_clicks'),
                       State('mask-dict', 'data'),
                       State('derive-cell-boundary', 'value'),
                       prevent_initial_call=True)
    def set_mask_dict_and_options(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict, derive_cell_boundary):
        # cases where the callback should occur: if the mask dict is longer than 1 and triggered by the dictionary
        # or, if there is a single mask and the trigger is setting the mask name
        multi_upload = ctx.triggered_id == "mask-uploads" and len(mask_uploads) > 1
        single_upload = ctx.triggered_id == 'set-mask-name' and len(mask_uploads) == 1
        if multi_upload or single_upload:
            return read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name, set_mask,
                                                cur_mask_dict, derive_cell_boundary)
        else:
            raise PreventUpdate

    # @dash_app.callback(
    #     Output("quantification-config-modal", "is_open"),
    #     Input('cell-type-col-designation', 'options'),
    #     prevent_initial_call=True)
    # def toggle_annotation_col_modal(quantification_dict):
    #     """
    #     Toggle the annotation modal on or off when the quantification dataset
    #     updates the possible cell type annotations
    #     """
    #     if quantification_dict is not None:
    #         return True
    #     else:
    #         return False

    @dash_app.callback(
        Input("annotations-dict", "data"),
        State('quantification-dict', 'data'),
        State('data-collection', 'value'),
        State('data-collection', 'options'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        Output('quantification-dict', 'data', allow_duplicate=True),
        Output("annotations-dict", "data", allow_duplicate=True))
    def add_region_annotation_to_quantification_frame(annotations, quantification_frame, data_selection,
                                                      data_dropdown_options, mask_config, mask_toggle, mask_selection):
        """
        Add a region annotation to the cells of a quantification data frame
        """
        sample_name, id_column = identify_column_matching_roi_to_quantification(
            data_selection, quantification_frame, data_dropdown_options)
        quant_frame, annotations = callback_add_region_annotation_to_quantification_frame(annotations,
                                quantification_frame, data_selection, mask_config, mask_toggle,
                                mask_selection, sample_name=sample_name, id_column=id_column)
        return Serverside(quant_frame), annotations


    @dash_app.callback(
        Output("download-edited-annotations", "data"),
        Input("btn-download-annotations", "n_clicks"),
        Input("quantification-dict", "data"))
    # @cache.memoize())
    def download_quantification_with_annotations(n_clicks, datatable_contents):
        if n_clicks is not None and n_clicks > 0 and datatable_contents is not None and \
                ctx.triggered_id == "btn-download-annotations":
            return dcc.send_data_frame(pd.DataFrame(datatable_contents).to_csv, "annotations.csv")
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("download-annotation-pdf", "data"),
        Input("btn-download-annot-pdf", "n_clicks"),
        State("annotations-dict", "data"),
        State('canvas-layers', 'data'),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('alias-dict', 'data'),
        State('blending_colours', 'data'))
    # @cache.memoize())
    def download_annotations_pdf(n_clicks, annotations_dict, canvas_layers,
                                 data_selection, mask_config, aliases, blend_dict):
        if n_clicks > 0 and None not in (annotations_dict, canvas_layers, data_selection):
            dest_path = os.path.join(tmpdirname, authentic_id, 'downloads')
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            return dcc.send_file(generate_annotations_output_pdf(annotations_dict, canvas_layers, data_selection,
                mask_config, aliases, blend_dict=blend_dict,
                dest_dir=dest_path, output_file="annotations.pdf"), type="application/pdf")
        else:
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
    # @cache.memoize())
    def download_annotations_masks(n_clicks, annotations_dict, canvas_layers,
                                 data_selection, image_dict, mask_dict, apply_mask, mask_selection):
        if n_clicks > 0 and None not in (annotations_dict, canvas_layers, data_selection, image_dict) and \
                data_selection in annotations_dict and len(annotations_dict[data_selection]) > 0:
            first_image = list(image_dict[data_selection].keys())[0]
            first_image = image_dict[data_selection][first_image]
            dest_path = os.path.join(tmpdirname, authentic_id, 'downloads', 'annotation_masks')
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            # check that the mask is compatible with the current image
            if None not in (mask_dict, mask_selection) and apply_mask and validate_mask_shape_matches_image(first_image,
                                                                                mask_dict[mask_selection]['raw']):
                mask_used = mask_dict[mask_selection]['raw']
            else:
                mask_used = None
            return dcc.send_file(export_annotations_as_masks(annotations_dict, dest_path, data_selection,
                                                             (first_image.shape[0], first_image.shape[1]),
                                                             mask_used))
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output('annotation_canvas', 'figure', allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Input("clear-region-annotation-shapes", "n_clicks"),
        State('annotation_canvas', 'figure'),
        State('annotation_canvas', 'relayoutData'),
        State('session_alert_config', 'data'))
    # @cache.memoize())
    def clear_canvas_shapes(n_clicks, cur_canvas, canvas_layout, error_config):
        """
        Clear the current canvas of any shapes that are not associated with the legend or scalebar
        Important: requires that the current dragmode be set to zoom or pan to remove any shapes in the current
        canvas layout
        """
        return callback_remove_canvas_annotation_shapes(n_clicks, cur_canvas, canvas_layout, error_config)

    @dash_app.callback(
        Output("annotations-dict", "data", allow_duplicate=True),
        Input('clear-annotation_dict', 'n_clicks'),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        prevent_initial_call=True)
    # @cache.memoize())
    def clear_current_roi_annotations(n_clicks, cur_annotation_dict, data_selection):
        """
        Clear all the current ROI annotations
        """
        if n_clicks > 0 and None not in (cur_annotation_dict, data_selection):
            try:
                cur_annotation_dict[data_selection] = {}
                return Serverside(cur_annotation_dict)
            except KeyError:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("show-quant-dist-table", "is_open"),
        Input('show-quant-dist', 'n_clicks'),
        [State("show-quant-dist-table", "is_open")])
    def toggle_show_quant_dist_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @dash_app.callback(Output('quant-dist-table', 'data'),
                       Output('quant-dist-table', 'columns'),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Input("umap-projection-options", "value"),
                       Input('quantification-dict', 'data'),
                       Input('cur-umap-subset-category-counts', 'data'),
                       prevent_initial_call=True)
    def populate_quantification_distribution_table(umap_variable, quantification_dict, subset_cur_cat):
        # TODO: populate the frequency distribution table for a variable in the quantification results
        if None not in (quantification_dict, umap_variable):
            if subset_cur_cat is None:
                frame = pd.DataFrame(quantification_dict)[umap_variable].value_counts().reset_index().rename(
                    columns={"index": "Value", 0: "Count"})
            else:
                frame = pd.DataFrame(zip(list(subset_cur_cat.keys()),
                                         list(subset_cur_cat.values())), columns=["Value", "Counts"])
            # frame.reset_index().rename(columns={"index": "Value", 0: "Count"})
            columns = [{'id': p, 'name': p, 'editable': False} for p in list(frame.columns)]
            return frame.to_dict(orient="records"), columns, dash.no_update
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("download-point-csv", "data"),
        Input("btn-download-points-csv", "n_clicks"),
        State("annotations-dict", "data"),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        State('uploaded_dict', 'data'),
        prevent_initial_call=True)
    # @cache.memoize())
    def download_point_annotations_as_csv(n_clicks, annotations_dict, data_selection,
                                          mask_dict, apply_mask, mask_selection, image_dict):
        exp, slide, acq = split_string_at_pattern(data_selection)
        return export_point_annotations_as_csv(n_clicks, acq, annotations_dict, data_selection,
                                          mask_dict, apply_mask, mask_selection, image_dict,
                                               authentic_id, tmpdirname)

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
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output("umap-config-modal", "is_open"),
        Input('umap-config-button', 'n_clicks'),
        [State("umap-config-modal", "is_open")])
    def toggle_show_umap_config_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open
