import dash

from .cell_level_wrappers import *
from ..parsers.roi_parsers import *
from ..io.gallery_outputs import *
import dash_bootstrap_components as dbc
from dash import html, ALL

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
                       prevent_initial_call=True)
    def generate_roi_images_from_query(currently_selected, data_selection, blend_colour_dict,
                                    session_config, execute_query, num_queries, rois_exclude, load_additional,
                                    existing_gallery, execute_quant_query, query_from_quantification):
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
            images = generate_multi_roi_images_from_query(data_selection, session_config, blend_colour_dict,
                                                    currently_selected, int(num_queries), rois_exclude, rois_decided)
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
