import dash_uploader as du
from dash_extensions.enrich import Output, Input, State
from dash import ctx
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *

def init_cell_level_callbacks(dash_app):
    """
    Initialize the callbacks associated with cell level analysis (object detection, quantification, dimensional reduction)
    """
    dash_app.config.suppress_callback_exceptions = True

    @du.callback(Output('session_config_quantification', 'data'),
                 id='upload-quantification')
    # @cache.memoize())
    def get_quantification_upload_from_drag_and_drop(status: du.UploadStatus):
        return get_quantification_filepaths_from_drag_and_drop(status)

    @dash_app.callback(Output('quantification-dict', 'data'),
                       Input('session_config_quantification', 'data'),
                       prevent_initial_call=True)
    def populate_quantification_table_from_upload(session_dict):
        return parse_and_validate_measurements_csv(session_dict)

    @dash_app.callback(Output('quantification-bar-full', 'figure'),
                       Input('quantification-dict', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('quantification-bar-mode', 'value'),
                       Input('umap-plot', 'relayoutData'),
                       State('umap-projection', 'data'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_statistics(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings):
        zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]','yaxis.range[0]', 'yaxis.range[1]']
        return generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, zoom_keys, ctx.triggered_id)

    @dash_app.callback(Output('umap-projection', 'data'),
                       Output('umap-projection-options', 'options'),
                       Input('quantification-dict', 'data'),
                       prevent_initial_call=True)
    def generate_umap_from_measurements_csv(quantification_dict):
        """
        Generate a umap data frame projection of the measurements csv quantification. Returns a data frame
        of the embeddings and a list of the channels for interactive projection
        """
        return return_umap_dataframe_from_quantification_dict(quantification_dict)

    @dash_app.callback(Output('umap-plot', 'figure'),
                       Input('umap-projection', 'data'),
                       Input('umap-projection-options', 'value'),
                       State('quantification-dict', 'data'),
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
        return send_alert_on_imcompatible_mask(mask_dict, data_selection, upload_dict, error_config, mask_selection,
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

    # @dash_app.callback(Output('umap-plot', 'figure'),
    #                    Input('anndata', 'data'),
    #                    Input('metadata_options', 'value'),
    #                    Input('dimension-reduction_options', 'value'))
    # # @cache.memoize())
    # def render_umap_plot(anndata_obj, metadata_selection, assay_selection):
    #     if anndata_obj and "assays" in anndata_obj.keys() and metadata_selection and assay_selection:
    #         umap_data = anndata_obj["full_obj"]
    #         return px.scatter(umap_data.obsm[assay_selection], x=0, y=1, color=umap_data.obs[metadata_selection],
    #                           labels={'color': metadata_selection})
    #     else:
    #         raise PreventUpdate

    # @dash_app.callback(
    #     Output("metadata-distribution", "figure"),
    #     Input('anndata', 'data'),
    #     Input('metadata_options', 'value'))
    # # @cache.memoize())
    # def display_metadata_distribution(anndata_obj, metadata_selection):
    #     if anndata_obj is not None and metadata_selection is not None:
    #         ann_data = anndata_obj['metadata'][metadata_selection]
    #         fig = px.histogram(ann_data, range_x=[min(ann_data), max(ann_data)])
    #         return fig
    #     else:
    #         raise PreventUpdate
    #
    # @du.callback(Output('anndata', 'data'),
    #              id='upload-quantification')
    # # @cache.memoize())
    # def create_layered_dict(status: du.UploadStatus):
    #     filenames = [str(x) for x in status.uploaded_files]
    #     anndata_files = {}
    #     if filenames:
    #         for data_file in filenames:
    #             anndata_dict = {}
    #             data = anndata.read_h5ad(data_file)
    #             anndata_dict["file_path"] = str(data_file)
    #             anndata_dict["observations"] = data.X
    #             anndata_dict["metadata"] = data.obs
    #             anndata_dict["full_obj"] = data
    #             for sub_assay in data.obsm_keys():
    #                 if "assays" not in anndata_dict.keys():
    #                     anndata_dict["assays"] = {sub_assay: data.obsm[sub_assay]}
    #                 else:
    #                     anndata_dict["assays"][sub_assay] = data.obsm[sub_assay]
    #             anndata_files = anndata_dict
    #     if anndata_files is not None and len(anndata_files) > 0:
    #         return Serverside(anndata_files)
    #     else:
    #         raise PreventUpdate
    #
    # @dash_app.callback(Output('dimension-reduction_options', 'options'),
    #                    Input('anndata', 'data'))
    # # @cache.memoize())
    # def create_anndata_dimension_options(anndata_dict):
    #     if anndata_dict and "assays" in anndata_dict.keys():
    #         return [{'label': i, 'value': i} for i in anndata_dict["assays"].keys()]
    #     else:
    #         raise PreventUpdate
    #
    # @dash_app.callback(Output('metadata_options', 'options'),
    #                    Input('anndata', 'data'))
    # # @cache.memoize())
    # def create_anndata_dimension_options(anndata_dict):
    #     if anndata_dict and "metadata" in anndata_dict.keys():
    #         return [{'label': i, 'value': i} for i in anndata_dict["metadata"].columns]
    #     else:
    #         raise PreventUpdate
