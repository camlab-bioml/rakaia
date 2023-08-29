import dash
import dash_uploader as du
import numpy as np
import pandas as pd
from dash_extensions.enrich import Output, Input, State
from dash import ctx
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *
from .cell_level_wrappers import *
from dash import dcc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

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
            split = split_string_at_pattern(data_selection)
            exp, slide, acq = split[0], split[1], split[2]
            first_image = list(upload_dict[exp][slide][acq].keys())[0]
            image_for_validation = upload_dict[exp][slide][acq][first_image]
        else:
            image_for_validation = None
        return parse_and_validate_measurements_csv(session_dict, error_config=error_config,
                                                   image_to_validate=image_for_validation)

    @dash_app.callback(Output('quantification-bar-full', 'figure'),
                       Input('quantification-dict', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('quantification-bar-mode', 'value'),
                       Input('umap-plot', 'relayoutData'),
                       State('umap-projection', 'data'),
                       State('quant-annotation-col', 'options'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_statistics(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, annot_cols):
        zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]','yaxis.range[0]', 'yaxis.range[1]']
        return generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, zoom_keys, ctx.triggered_id, annot_cols)

    @dash_app.callback(Output('umap-projection', 'data'),
                       Output('umap-projection-options', 'options'),
                       Input('quantification-dict', 'data'),
                       State('umap-projection', 'data'),
                       prevent_initial_call=True)
    def generate_umap_from_measurements_csv(quantification_dict, current_umap):
        """
        Generate a umap data frame projection of the measurements csv quantification. Returns a data frame
        of the embeddings and a list of the channels for interactive projection
        """
        try:
            return return_umap_dataframe_from_quantification_dict(quantification_dict=quantification_dict,
                                                                  current_umap=current_umap)
        except ValueError:
            return dash.no_update, list(pd.DataFrame(quantification_dict).columns)

    @dash_app.callback(Output('umap-plot', 'figure'),
                       Input('umap-projection', 'data'),
                       Input('umap-projection-options', 'value'),
                       Input('quantification-dict', 'data'),
                       State('umap-plot', 'figure'),
                       prevent_initial_call=True)
    def plot_umap_for_measurements(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
        return generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig)

    @du.callback(Output('mask-uploads', 'data'),
                 id='upload-mask')
    # @cache.memoize())
    def return_mask_upload(status: du.UploadStatus):
        return parse_masks_from_filenames(status)

    @dash_app.callback(Output('input-mask-name', 'value'),
                       Input('mask-uploads', 'data'),
                       prevent_initial_call=True)
    def input_mask_name_on_upload(mask_uploads):
        if mask_uploads is not None and len(mask_uploads) > 0:
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
        Input('set-mask-name', 'n_clicks'))
    def toggle_mask_name_input_modal(new_mask_name, mask_clicks):
        if new_mask_name and ctx.triggered_id == "input-mask-name":
            return True
        elif ctx.triggered_id == "set-mask-name" and mask_clicks > 0:
            return False
        else:
            return False

    @dash_app.callback(Output('mask-dict', 'data'),
                       Output('mask-options', 'options'),
                       State('mask-uploads', 'data'),
                       State('input-mask-name', 'value'),
                       Input('set-mask-name', 'n_clicks'),
                       State('mask-dict', 'data'),
                       State('derive-cell-boundary', 'value'),
                       prevent_initial_call=True)
    def set_mask_dict_and_options(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict, derive_cell_boundary):
        return read_in_mask_array_from_filepath(mask_uploads, chosen_mask_name, set_mask,
                                                cur_mask_dict, derive_cell_boundary)

    @dash_app.callback(
        Output("quantification-config-modal", "is_open"),
        Input('cell-type-col-designation', 'options'),
        prevent_initial_call=True)
    def toggle_annotation_col_modal(quantification_dict):
        """
        Toggle the annotation modal on or off when the quantification dataset
        updates the possible cell type annotations
        """
        if quantification_dict is not None:
            return True
        else:
            return False

    @dash_app.callback(
        Input("annotations-dict", "data"),
        State('quantification-dict', 'data'),
        State('data-collection', 'value'),
        State('mask-dict', 'data'),
        State('apply-mask', 'value'),
        State('mask-options', 'value'),
        Output('quantification-dict', 'data', allow_duplicate=True),
        Output("annotations-dict", "data", allow_duplicate=True))
    def add_region_annotation_to_quantification_frame(annotations, quantification_frame, data_selection,
                                                      mask_config, mask_toggle, mask_selection):
        """
        Add a region annotation to the cells of a quantification data frame
        """
        # loop through all of the existing annotations
        # for annotations that have not yet been imported, import and set the import status to True
        return callback_add_region_annotation_to_quantification_frame(annotations, quantification_frame, data_selection,
                                                      mask_config, mask_toggle, mask_selection)


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
        State('alias-dict', 'data'),)
    # @cache.memoize())
    def download_annotations_pdf(n_clicks, annotations_dict, canvas_layers, data_selection, mask_config, aliases):
        if n_clicks > 0 and None not in (annotations_dict, canvas_layers, data_selection):
            dest_path = os.path.join(tmpdirname, authentic_id, 'downloads')
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            return dcc.send_file(generate_annotations_output_pdf(annotations_dict, canvas_layers, data_selection,
                mask_config, aliases, dest_dir=dest_path, output_file="annotations.pdf"), type="application/pdf")
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
                       prevent_initial_call=True)
    def populate_quantification_distribution_table(umap_variable, quantification_dict):
        # TODO: populate the frequency distribution table for a variable in the quantification results
        if None not in (quantification_dict, umap_variable):
            frame = pd.DataFrame(quantification_dict)[umap_variable].value_counts().reset_index().rename(
                columns={"index": "Value", 0: "Count"})
            columns = [{'id': p, 'name': p, 'editable': False} for p in list(frame.columns)]
            return frame.to_dict(orient="records"), columns, dash.no_update
        else:
            raise PreventUpdate
