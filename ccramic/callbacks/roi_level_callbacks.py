import dash
import pandas as pd

from ccramic.parsers.roi_parsers import generate_multi_roi_images_from_query
from ccramic.io.gallery_outputs import generate_roi_query_gallery_children
from ccramic.utils.quantification import (
    quantify_multiple_channels_per_roi,
    concat_quantification_frames_multi_roi)
from ccramic.utils.cell_level_utils import validate_mask_shape_matches_image
from dash import ALL
from dash_extensions.enrich import Output, State, Input
from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Serverside

def init_roi_level_callbacks(dash_app, tmpdirname, authentic_id):
    """
    Initialize the callbacks associated with ROI level and cross dataset queries
    """

    @dash_app.callback(Output('dataset-query-gallery-row', 'children'),
                       Output('roi-query', 'data'),
                       Output('dataset-query-gallery', 'style'),
                       Output('dataset-query-gallery-list', 'data'),
                       Output('pixel-level-analysis', 'active_tab', allow_duplicate=True),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       State('session_config', 'data'),
                       Input('execute-dataset-query', 'n_clicks'),
                       State('dataset-query-number', 'value'),
                       State('dataset-query-gallery-list', 'data'),
                       Input('dataset-query-additional-load', "n_clicks"),
                       State('dataset-query-gallery-row', 'children'),
                       Input('quantification-query-link', 'n_clicks'),
                       State('quantification-query-indices', 'data'),
                       State('mask-dict', 'data'),
                       State('data-collection', 'options'),
                       State('query-cell-id-lists', 'data'),
                       prevent_initial_call=True)
    def generate_roi_images_from_query(currently_selected, data_selection, blend_colour_dict,
                                    session_config, execute_query, num_queries, rois_exclude, load_additional,
                                    existing_gallery, execute_quant_query, query_from_quantification, mask_dict,
                                    dataset_options, query_cell_id_lists):
        """
        Generate the dynamic gallery of ROI queries from the query selection
        Can be activated using either the original button for a fresh query, or the button to load additional ROIs
        on top of the current gallery
        """
        # do not execute query if triggered from the quantification tab and no sample indices exist
        quant_empty = ctx.triggered_id == "quantification-query-link" and query_from_quantification is None
        if None not in (currently_selected, data_selection, blend_colour_dict,
                        session_config) and not quant_empty and len(currently_selected) > 0:
            if ctx.triggered_id == "quantification-query-link" and execute_quant_query > 0:
                rois_decided = query_from_quantification
                rois_exclude = []
                row_children = []
            else:
                rois_decided = None
            # if the query is being extended, append on top of the existing gallery
            if ctx.triggered_id == "dataset-query-additional-load" and load_additional > 0:
                rois_exclude = rois_exclude
                row_children = existing_gallery
            elif ctx.triggered_id == "execute-dataset-query" and execute_query > 0:
                rois_exclude = [data_selection]
                row_children = []
            images = generate_multi_roi_images_from_query(session_config, blend_colour_dict,
                    currently_selected, int(num_queries), rois_exclude, rois_decided, mask_dict, dataset_options,
                                                          query_cell_id_lists)
            new_row_children, roi_list = generate_roi_query_gallery_children(images)
            # if the query is being extended, append to the existing gallery for exclusion. Otherwise, start fresh
            if ctx.triggered_id == "dataset-query-additional-load":
                roi_list = list(set(rois_exclude + roi_list))
            roi_list.append(data_selection)
            row_children = row_children + new_row_children
            return row_children, num_queries, {"margin-top": "15px", "display": "block"}, roi_list, "dataset-query"
        else:
            raise PreventUpdate


    @dash_app.callback(
        Output('data-collection', 'value', allow_duplicate=True),
        Output('pixel-level-analysis', 'active_tab', allow_duplicate=True),
        Input({'type': 'data-query-gallery', "index": ALL}, "n_clicks"),
        State('data-collection', 'options'),
        State('data-collection', 'value'),
        prevent_initial_call=True)
    # @cache.memoize())
    def load_roi_through_query_click(roi_query, dataset_options, current_roi):
        if dataset_options is not None and not all([elem is None for elem in roi_query]):
            index_from = ctx.triggered_id["index"]
            if index_from in dataset_options and index_from != current_roi:
                return index_from, "pixel-analysis"
            else:
                raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output('quantification-dict', 'data', allow_duplicate=True),
        Output('session_alert_config', 'data', allow_duplicate=True),
        Output('umap-div-holder', 'style', allow_duplicate=True),
        Output('umap-projection-options', 'value', allow_duplicate=True),
        Input('quantify-cur-roi-execute', 'n_clicks'),
        State('apply-mask', 'value'),
        State('mask-dict', 'data'),
        State('mask-options', 'value'),
        State('uploaded_dict', 'data'),
        State('data-collection', 'value'),
        State('data-collection', 'options'),
        State('quantification-dict', 'data'),
        State('channel-quantification-list', 'value'),
        State('alias-dict', 'data'),
        State('session_alert_config', 'data'),
        prevent_initial_call=True)
    # @cache.memoize())
    def quantify_current_roi(execute, apply_mask, mask_dict, mask_selection, image_dict, data_selection,
                             dataset_options, cur_quant_dict, channels_to_quantify, aliases, error_config):
        """
        Quantify the current ROI using the currently applied mask
        Important: the UMAP figure and UMAP annotation column are both reset when new quantification results are
        obtained as the UMAP projections will no longer align with the quantification frame and must be re-run
        If the quantification is successful, close the modal
        """
        if error_config is None:
            error_config = {"error": None}
        if execute > 0 and None not in (image_dict, data_selection, mask_selection, channels_to_quantify) and \
                apply_mask:
            first_image = list(image_dict[data_selection].keys())[0]
            first_image = image_dict[data_selection][first_image]
            if validate_mask_shape_matches_image(first_image, mask_dict[mask_selection]['raw']):
                new_quant = quantify_multiple_channels_per_roi(image_dict, mask_dict[mask_selection]['raw'],
                                            data_selection, channels_to_quantify, aliases, dataset_options)
                quant_frame = concat_quantification_frames_multi_roi(pd.DataFrame(cur_quant_dict), new_quant,
                                                                     data_selection)
                return Serverside(quant_frame.to_dict(orient="records")), dash.no_update, {'display': 'None'}, None
            else:
                error_config["error"] = "The dimensions of the mask do not agree with the current ROI."
                return dash.no_update, error_config, dash.no_update, dash.no_update
        else:
            error_config["error"] = "Quantification requires an ROI with a compatible mask that has been applied to the" \
                                    " canvas. Please review the required inputs."
            return dash.no_update, error_config, dash.no_update, dash.no_update
