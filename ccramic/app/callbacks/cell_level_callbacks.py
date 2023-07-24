import pandas as pd
from dash.exceptions import PreventUpdate
import dash_uploader as du
from dash_extensions.enrich import Output, Input, State, Serverside
from dash import ctx
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *
from tifffile import TiffFile
import os
from sklearn.preprocessing import StandardScaler
import sys

def init_cell_level_callbacks(dash_app):
    """
    Initialize the callbacks associated with cell level analysis (object detection, quantification, dimensional reduction)
    """
    dash_app.config.suppress_callback_exceptions = True

    @du.callback(Output('session_config_quantification', 'data'),
                 id='upload-quantification')
    # @cache.memoize())
    def get_quantification_upload_from_drag_and_drop(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        session_config = {'uploads': []}
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if filenames and float(status.progress) == 1.0:
            for file in filenames:
                session_config['uploads'].append(file)
            return session_config
        else:
            raise PreventUpdate

    @dash_app.callback(Output('quantification-dict', 'data'),
                       Input('session_config_quantification', 'data'),
                       prevent_initial_call=True)
    def populate_quantification_table_from_upload(session_dict):
        if session_dict is not None and 'uploads' in session_dict.keys() and len(session_dict['uploads']) > 0:
            quantification_worksheet = validate_incoming_measurements_csv(pd.read_csv(session_dict['uploads'][0]),
                                                         validate_with_image=False)
            # TODO: establish where to use the percentile filtering on the measurements
            return filter_measurements_csv_by_channel_percentile(quantification_worksheet).to_dict(orient="records")
        else:
            raise PreventUpdate

    @dash_app.callback(Output('quantification-bar-full', 'figure'),
                       Input('quantification-dict', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('quantification-bar-mode', 'value'),
                       Input('umap-plot', 'relayoutData'),
                       State('umap-projection', 'data'),
                       State('quantification-bar-full', 'figure'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_statistics(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, cur_quantification_graph):
        zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]','yaxis.range[0]', 'yaxis.range[1]']
        if quantification_dict is not None and len(quantification_dict) > 0:
            if all([key in canvas_layout for key in zoom_keys]) and ctx.triggered_id == "annotation_canvas":
                subset_zoom = {"x_min": min(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                               "x_max": max(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                               "y_min": min(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]']),
                               "y_max": max(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]'])}
                fig = go.Figure(get_cell_channel_expression_plot(pd.DataFrame(quantification_dict),
                                                        subset_dict=subset_zoom, mode=mode_value))
            elif ctx.triggered_id == "umap-plot" and all([key in umap_layout for key in zoom_keys]):
                subset_frame = subset_measurements_frame_from_umap_coordinates(pd.DataFrame(quantification_dict),
                                                    pd.DataFrame(embeddings, columns = ['UMAP1', 'UMAP2']),
                                                                               umap_layout)
                fig = go.Figure(get_cell_channel_expression_plot(subset_frame,
                                                        subset_dict=None, mode=mode_value))
            else:
                subset_zoom=None
                fig = go.Figure(get_cell_channel_expression_plot(pd.DataFrame(quantification_dict),
                                                        subset_dict=subset_zoom, mode=mode_value))
            # print(cur_quantification_graph)
            # if cur_quantification_graph is not None and 'layout' in cur_quantification_graph and \
            #     'uirevision' in cur_quantification_graph['layout'] and \
            #         cur_quantification_graph['layout']['uirevision']:
            #     fig['layout'] = cur_quantification_graph['layout']
            # else:
            fig['layout']['uirevision'] = True
            return fig
        else:
            raise PreventUpdate

    @dash_app.callback(Output('umap-projection', 'data'),
                       Output('umap-projection-options', 'options'),
                       Input('quantification-dict', 'data'),
                       prevent_initial_call=True)
    def generate_umap_from_measurements_csv(quantification_dict):
        """
        Generate a umap data frame projection of the measurements csv quantification. Returns a data frame
        of the embeddings and a list of the channels for interactive projection
        """
        if quantification_dict is not None:
            # TODO: process quantification by removing cells outside of the percentile range for pixel intensity (
            #  column-wise, by channel)
            data_frame = pd.DataFrame(quantification_dict)
            umap_obj = None
            for elem in ['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample']:
                if elem in data_frame.columns:
                    data_frame = data_frame.drop([elem], axis=1)
            # TODO: evaluate the umap import speed (slow) possibly due to numba compilation:
            # https://github.com/lmcinnes/umap/issues/631
            if 'umap' not in sys.modules:
                import umap
            try:
                umap_obj = umap.UMAP()
            except UnboundLocalError:
                import umap
                umap_obj = umap.UMAP()
            if umap_obj is not None:
                scaled = StandardScaler().fit_transform(data_frame)
                embedding = umap_obj.fit_transform(scaled)
                return Serverside(embedding), list(data_frame.columns)
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @dash_app.callback(Output('umap-plot', 'figure'),
                       Input('umap-projection', 'data'),
                       Input('umap-projection-options', 'value'),
                       State('quantification-dict', 'data'),
                       State('umap-plot', 'figure'),
                       prevent_initial_call=True)
    def plot_umap_for_measurements(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
        if embeddings is not None:
            quant_frame = pd.DataFrame(quantification_dict)
            colour = quant_frame[channel_overlay].astype(float) if channel_overlay is not None else None
            df = pd.DataFrame(embeddings, columns = ['UMAP1', 'UMAP2'])
            fig = px.scatter(df, x="UMAP1", y="UMAP2", color=colour)
            if cur_umap_fig is None:
                fig['layout']['uirevision'] = True
            else:
                fig['layout'] = cur_umap_fig['layout']
            return fig
        else:
            raise PreventUpdate



    @du.callback(Output('mask-uploads', 'data'),
                 id='upload-mask')
    # @cache.memoize())
    def return_mask_upload(status: du.UploadStatus):
        filenames = [str(x) for x in status.uploaded_files]
        mask_config = {'mask': []}
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if len(filenames) == 1:
            default_mask_name = os.path.splitext(os.path.basename(filenames[0]))[0]
            return {default_mask_name: filenames[0]}
        else:
            raise PreventUpdate

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
        if None not in (mask_dict, data_selection, upload_dict, mask_selection) and mask_toggle:
            split = split_string_at_pattern(data_selection)
            exp, slide, acq = split[0], split[1], split[2]
            first_image = list(upload_dict[exp][slide][acq].keys())[0]
            first_image = upload_dict[exp][slide][acq][first_image]
            if first_image.shape[0] != mask_dict[mask_selection].shape[0] or \
                    first_image.shape[1] != mask_dict[mask_selection].shape[1]:
                if error_config is None:
                    error_config = {"error": None}
                error_config["error"] = "Warning: the current mask does not have " \
                                        "the same dimensions as the current ROI."
                return error_config
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate

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
        if set_mask > 0 and None not in (mask_uploads, chosen_mask_name):
            cur_mask_dict = {} if cur_mask_dict is None else cur_mask_dict
            with TiffFile(str(mask_uploads[list(mask_uploads.keys())[0]])) as tif:
                for page in tif.pages:
                    if derive_cell_boundary:
                        mask_import = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(page.asarray())).convert('RGB'))
                    else:
                        mask_import = np.array(Image.fromarray(page.asarray()).convert('RGB'))
                    cur_mask_dict[chosen_mask_name] = mask_import
            return Serverside(cur_mask_dict), list(cur_mask_dict.keys())
        else:
            raise PreventUpdate



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
