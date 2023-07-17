
from dash.exceptions import PreventUpdate
import dash_uploader as du
from dash_extensions.enrich import Output, Input, State, Serverside
from dash import ctx
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *
from tifffile import TiffFile
import os
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
                                                         validate_with_image=False).to_dict(orient="records")
            return quantification_worksheet
        else:
            raise PreventUpdate

    @dash_app.callback(Output('quantification-bar-full', 'figure'),
                       Input('quantification-dict', 'data'),
                       Input('annotation_canvas', 'relayoutData'),
                       Input('quantification-bar-mode', 'value'),
                       prevent_initial_call=True)
    def get_cell_channel_expression_statistics(quantification_dict, canvas_layout, mode_value):
        zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]','yaxis.range[0]', 'yaxis.range[1]']
        if quantification_dict is not None and len(quantification_dict) > 0:
            if all([key in canvas_layout for key in zoom_keys]):
                subset_zoom = {"x_min": min(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                               "x_max": max(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                               "y_min": min(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]']),
                               "y_max": max(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]'])}
            else:
                subset_zoom=None
            return get_cell_channel_expression_plot(pd.DataFrame(quantification_dict),
                                                    subset_dict=subset_zoom, mode=mode_value)
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
                       prevent_initial_call=True)
    def set_mask_dict_and_options(mask_uploads, chosen_mask_name, set_mask, cur_mask_dict):
        if set_mask > 0 and None not in (mask_uploads, chosen_mask_name):
            cur_mask_dict = {} if cur_mask_dict is None else cur_mask_dict
            with TiffFile(str(mask_uploads[list(mask_uploads.keys())[0]])) as tif:
                for page in tif.pages:
                    cur_mask_dict[chosen_mask_name] = np.array(Image.fromarray(
                        convert_mask_to_cell_boundary(page.asarray())).convert('RGB'))
            return Serverside(cur_mask_dict), list(cur_mask_dict.keys())
        else:
            raise PreventUpdate

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
