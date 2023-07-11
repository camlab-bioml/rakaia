import dash.exceptions
import pandas as pd
from dash.exceptions import PreventUpdate
import flask
import dash_uploader as du
from dash_extensions.enrich import Output, Input, State, Serverside, html
import dash_bootstrap_components as dbc
from dash import ctx
from tifffile import imwrite
import math
from numpy.core._exceptions import _ArrayMemoryError
from plotly.graph_objs.layout import XAxis, YAxis
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
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
