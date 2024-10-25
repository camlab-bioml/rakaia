import os
import uuid

import dash
import pandas as pd
from dash import ALL, dcc
from dash_extensions.enrich import Output, State, Input
from dash import ctx
from dash.exceptions import PreventUpdate
from rakaia.parsers.roi import RegionThumbnail
from rakaia.io.gallery import (
            roi_query_gallery_children,
            gallery_export_template,
            channel_tiles_from_gallery)
from rakaia.utils.decorator import DownloadDirGenerator
from rakaia.utils.pixel import get_region_dim_from_roi_dictionary
from rakaia.utils.quantification import (
    quantify_multiple_channels_per_roi,
    concat_quantification_frames_multi_roi)
from rakaia.utils.object import (
    validate_mask_shape_matches_image,
    ROIQuantificationMatch,
    find_similar_images)
from rakaia.utils.alert import AlertMessage, add_warning_to_error_config
from rakaia.io.session import SessionServerside
from rakaia.utils.roi import override_roi_gallery_blend_list


def init_roi_level_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with ROI level and cross dataset queries

    :param dash_app: the dash proxy server wrapped in the parent Flask app
    :param tmpdirname: the path for the tmpdir for tmp storage for the session
    :param authentic_id: uuid string identifying the particular app invocation
    :param app_config: Dictionary of session options passed through CLI
    :return: None
    """
    @dash_app.callback(Output('dataset-query-gallery-row', 'children'),
                       Output('roi-query', 'data'),
                       Output('dataset-query-gallery', 'style'),
                       Output('dataset-query-gallery-list', 'data'),
                       Output('main-tabs', 'active_tab', allow_duplicate=True),
                       Output('session_alert_config', 'data', allow_duplicate=True),
                       Output('download-roi-tiles', 'data'),
                       Input('btn-download-roi-tiles', 'n_clicks'),
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
                       State('bool-apply-global-filter', 'value'),
                       State('global-filter-type', 'value'),
                       State("global-kernel-val-filter", 'value'),
                       State("global-sigma-val-filter", 'value'),
                       State('dataset-delimiter', 'value'),
                       State('session_alert_config', 'data'),
                       State('dataset-query-dim-min', 'value'),
                       State('dataset-query-dim-max', 'value'),
                       State('dataset-query-keyw', 'value'),
                       Input('saved-blend-options-roi', 'value'),
                       State('saved-blends', 'data'),
                       Input('find-similar', 'n_clicks'),
                       State('image-prioritization-cor', 'data'),
                       State('quantification-dict', 'data'),
                       State('spatial-spot-rad', 'value'),
                       prevent_initial_call=True)
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def generate_roi_images_from_query(export_roi, currently_selected, data_selection, blend_colour_dict,
                                    session_config, execute_query, num_queries, rois_exclude, load_additional,
                                    existing_gallery, execute_quant_query, query_from_quantification, mask_dict,
                                    dataset_options, query_cell_id_lists, global_apply_filter,
                                    global_filter_type, global_filter_val, global_filter_sigma, delimiter, error_config,
                                    dim_min, dim_max, keyw, saved_blend, saved_blend_dict, find_similar, image_cor, quant,
                                    spatial_rad):
        """
        Generate the dynamic gallery of ROI queries from the query selection
        Can be activated using either the original button for a fresh query, or the button to load additional ROIs
        on top of the current gallery
        """
        # do not execute query if triggered from the quantification tab and no sample indices exist
        quant_empty = ctx.triggered_id == "quantification-query-link" and query_from_quantification is None
        no_similarity_scores = ctx.triggered_id == "find-similar" and pd.DataFrame(image_cor).empty
        if ctx.triggered_id == "btn-download-roi-tiles" and existing_gallery:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dcc.send_file(gallery_export_template(os.path.join(export_roi, 'rois.html'),
                            channel_tiles_from_gallery(existing_gallery), 3)))
        elif None not in (currently_selected, data_selection, blend_colour_dict, session_config) and not \
            quant_empty and len(currently_selected) > 0 and not no_similarity_scores and ctx.triggered_id != "btn-download-roi-tiles":
            if ctx.triggered_id == "quantification-query-link" and execute_quant_query > 0:
                rois_decided, rois_exclude, row_children = query_from_quantification, [], []
            elif ctx.triggered_id == "find-similar" and quant is not None and find_similar:
                name, col = ROIQuantificationMatch(data_selection, quant, dataset_options, delimiter).get_matches()
                rois_decided = find_similar_images(image_cor, name, num_queries, col) if name else None
                # do not use object id lists if looking for similar images (focus on whole image not subsets)
                rois_exclude, row_children, query_cell_id_lists = [], [], None
            else:
                rois_decided, row_children = None, None
            # if the query is being extended, append on top of the existing gallery
            if ctx.triggered_id == "dataset-query-additional-load" and load_additional > 0:
                # cant use existing gallery, need to somehow get the names in the current gallery and remake
                rois_exclude, row_children = rois_exclude, existing_gallery
            elif ctx.triggered_id in ["execute-dataset-query", "saved-blend-options-roi"] and execute_query > 0:
                rois_exclude, row_children = [data_selection], []
            currently_selected = override_roi_gallery_blend_list(currently_selected, saved_blend_dict, saved_blend)
            images = RegionThumbnail(session_config, blend_colour_dict, currently_selected, int(num_queries), rois_exclude,
            rois_decided, mask_dict, dataset_options, query_cell_id_lists, global_apply_filter, global_filter_type, global_filter_val,
            global_filter_sigma, delimiter, False, dim_min, dim_max, keyw, False, spatial_rad).get_image_dict()

            new_row_children, roi_list = roi_query_gallery_children(images)
            # if the query is being extended, append to the existing gallery for exclusion. Otherwise, start fresh
            if ctx.triggered_id == "dataset-query-additional-load": roi_list = list(set(rois_exclude + roi_list))
            roi_list.append(data_selection)
            row_children = row_children + new_row_children if row_children else new_row_children
            return row_children, num_queries, {"margin-top": "15px", "display": "block"}, roi_list, "dataset-query", dash.no_update, dash.no_update
        error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["invalid_query"])
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, error_config, dash.no_update

    @dash_app.callback(
        Output('data-collection', 'value', allow_duplicate=True),
        Output('main-tabs', 'active_tab', allow_duplicate=True),
        Input({'type': 'data-query-gallery', "index": ALL}, "n_clicks"),
        State('data-collection', 'options'),
        State('data-collection', 'value'),
        prevent_initial_call=True)
    def load_roi_through_query_click(roi_query, dataset_options, current_roi):
        if dataset_options is not None and not all([elem is None for elem in roi_query]):
            index_from = ctx.triggered_id["index"]
            if index_from in dataset_options and index_from != current_roi:
                return index_from, "pixel-analysis"
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
        State('dataset-delimiter', 'value'),
        prevent_initial_call=True)
    def quantify_current_roi(execute, apply_mask, mask_dict, mask_selection, image_dict, data_selection,
                             dataset_options, cur_quant_dict, channels_to_quantify, aliases, error_config, delimiter):
        """
        Quantify the current ROI using the currently applied mask
        Important: the UMAP figure and UMAP annotation column are both reset when new quantification results are
        obtained as the UMAP projections will no longer align with the quantification frame and must be re-run
        If the quantification is successful, close the modal
        """
        if None not in (image_dict, data_selection, mask_selection) and apply_mask and channels_to_quantify:
            first_image = get_region_dim_from_roi_dictionary(image_dict[data_selection])
            if validate_mask_shape_matches_image(first_image, mask_dict[mask_selection]['raw']):
                new_quant = quantify_multiple_channels_per_roi(image_dict, mask_dict[mask_selection]['raw'],
                            data_selection, channels_to_quantify, aliases, dataset_options, delimiter, mask_selection)
                quant_frame = concat_quantification_frames_multi_roi(pd.DataFrame(cur_quant_dict), new_quant, data_selection, delimiter)
                return SessionServerside(quant_frame.to_dict(orient="records"), key="quantification_dict",
                        use_unique_key=app_config['serverside_overwrite']), dash.no_update, {'display': 'None'}, None
            else:
                error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["invalid_dimensions"])
                return dash.no_update, error_config, dash.no_update, dash.no_update
        else:
            error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["quantification_missing"])
            return dash.no_update, error_config, dash.no_update, dash.no_update
