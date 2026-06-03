"""Application callbacks associated with region-level operations
(quantification, region querying)"""

import os
import uuid
import dash
import pandas as pd
from dash import ALL, dcc, html
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, State, Input
from dash import ctx

from rakaia.callbacks.triggers import stitch_cache_delete
from rakaia.stitch.cosmx import cosmx_global_slide_boundaries, cosmx_local_fov_position
from rakaia.stitch.gallery import ROIGalleryStitchParser
from rakaia.stitch.mcd import (
    MCDAcqCoordinateParser)
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
from rakaia.stitch import stitch_cache_dropdown_labels, download_stitch_image, stitch_image_preview, modify_stitch_cache
from rakaia.utils.session import roi_from_anndata_file


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
                       Output('roi_gallery_allow_click', 'data'),
                       Output('stitched_images', 'data'),
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
                       State('saved-blend-options-roi', 'value'),
                       State('saved-blends', 'data'),
                       Input('find-similar', 'n_clicks'),
                       State('image-prioritization-cor', 'data'),
                       State('quantification-dict', 'data'),
                       State('spatial-spot-rad', 'value'),
                       State('mask-in-gallery', 'value'),
                       State('query-min-obj', 'value'),
                       State('subsample-roi-gallery', 'value'),
                       Input('stitch-from-roi-gallery', 'n_clicks'),
                       State('session_id_internal', 'data'),
                       State('stitched_images', 'data'),
                       State('stitch-image-select', 'value'),
                       prevent_initial_call=True)
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id, str(uuid.uuid1()), 'downloads'))
    def generate_roi_images_from_query(export_roi, currently_selected, data_selection, blend_colour_dict,
                                    session_config, execute_query, num_queries, rois_exclude, load_additional,
                                    existing_gallery, execute_quant_query, query_from_quantification, mask_dict,
                                    dataset_options, query_cell_id_lists, global_apply_filter,
                                    global_filter_type, global_filter_val, global_filter_sigma, delimiter, error_config,
                                    dim_min, dim_max, keyw, saved_blend, saved_blend_dict, find_similar, image_cor, quant,
                                    spatial_rad, enable_masks, query_min, subsample_thumbnail, stitch_roi_tiles,
                                    sesh_id, stitch_cache, stitch_selection):
        """
        Generate the dynamic gallery of ROI queries from the query selection
        Can be activated using either the original button for a fresh query, or the button to load additional ROIs
        on top of the current gallery
        """
        # do not execute query if triggered from the quantification tab and no sample indices exist
        quant_empty = ctx.triggered_id == "quantification-query-link" and query_from_quantification is None
        no_similarity_scores = ctx.triggered_id == "find-similar" and pd.DataFrame(image_cor).empty
        allow_click, gallery_parser = True, None
        nothing_to_stitch = ctx.triggered_id == "stitch-from-roi-gallery" and not existing_gallery
        if ctx.triggered_id == "btn-download-roi-tiles" and existing_gallery:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dcc.send_file(gallery_export_template(os.path.join(export_roi, 'rois.html'),
                            channel_tiles_from_gallery(existing_gallery), 3)), allow_click, dash.no_update)
        elif (None not in (currently_selected, data_selection, blend_colour_dict, session_config) and not quant_empty and
              not nothing_to_stitch and len(currently_selected) > 0 and not no_similarity_scores and ctx.triggered_id != "btn-download-roi-tiles"):
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
                rois_exclude, row_children, allow_click = rois_exclude, existing_gallery, False
            elif ctx.triggered_id in ["execute-dataset-query", "saved-blend-options-roi"] and execute_query > 0:
                rois_exclude, row_children = [data_selection], []
            if ctx.triggered_id == "stitch-from-roi-gallery" and existing_gallery:
                # IMP: need to make sure that we keep the same rois that are currently in view
                # current best way to do this is to set the current ones in the gallery to search indices (which are normally excluded)
                gallery_parser = ROIGalleryStitchParser(stitch_cache, stitch_selection, rois_exclude, session_config, data_selection, delimiter)
                rois_decided, query_cell_id_lists, rois_exclude = gallery_parser.get_gallery_identifiers()
            currently_selected = override_roi_gallery_blend_list(currently_selected, saved_blend_dict, saved_blend)
            images = RegionThumbnail(session_config, blend_colour_dict, currently_selected, int(num_queries), rois_exclude, rois_decided,
            mask_dict, dataset_options, query_cell_id_lists, global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma,
            delimiter, False, dim_min, dim_max, keyw, False, spatial_rad, enable_masks, True,
                    app_config['array_store_type'], query_min).get_image_dict()
            # for stitching, once the images are remade, apply each of them to the stitch
            if ctx.triggered_id == "stitch-from-roi-gallery" and existing_gallery: return (tuple([dash.no_update] * 8) +
            tuple([SessionServerside(gallery_parser.update_stitch_from_gallery_thumbnails(images),
                    key=f"stitch_cache_{sesh_id}", use_unique_key=app_config['serverside_overwrite'])]))
            new_row_children, roi_list = roi_query_gallery_children(images, subsample_thumbnail=subsample_thumbnail)
            # if the query is being extended, append to the existing gallery for exclusion. Otherwise, start fresh
            if ctx.triggered_id == "dataset-query-additional-load": roi_list = list(set(rois_exclude + roi_list))
            roi_list.append(data_selection)
            row_children = row_children + new_row_children if row_children else new_row_children
            return row_children, num_queries, {"margin-top": "15px", "display": "block"}, roi_list, "dataset-query", dash.no_update, dash.no_update, allow_click, dash.no_update
        error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["invalid_query"])
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, error_config, dash.no_update, dash.no_update, dash.no_update

    @dash_app.callback(
        Output('data-collection', 'value', allow_duplicate=True),
        Output('main-tabs', 'active_tab', allow_duplicate=True),
        Output('roi_gallery_allow_click', 'data', allow_duplicate=True),
        Input({'type': 'data-query-gallery', "index": ALL}, "n_clicks"),
        State('data-collection', 'options'),
        State('data-collection', 'value'),
        State('roi_gallery_allow_click', 'data'),
        prevent_initial_call=True)
    def load_roi_through_query_click(roi_query, dataset_options, current_roi, allow_click):
        if dataset_options is not None and not all([elem is None for elem in roi_query]) and allow_click:
            index_from = ctx.triggered_id["index"]
            if index_from in dataset_options and index_from != current_roi:
                return index_from, "pixel-analysis", True
            return dash.no_update, dash.no_update, True
        return dash.no_update, dash.no_update, True

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
        State('session_id_internal', 'data'),
        prevent_initial_call=True)
    def quantify_current_roi(execute, apply_mask, mask_dict, mask_selection, image_dict, data_selection,
        dataset_options, cur_quant_dict, channels_to_quantify, aliases, error_config, delimiter, sesh_id):
        """
        Quantify the current ROI using the currently applied mask
        Important: the UMAP figure and UMAP annotation column are both reset when new quantification results are
        obtained as the UMAP projections will no longer align with the quantification frame and must be re-run
        If the quantification is successful, close the modal
        """
        if None not in (image_dict, data_selection, mask_selection) and apply_mask and channels_to_quantify and sesh_id:
            first_image = get_region_dim_from_roi_dictionary(image_dict[data_selection])
            if validate_mask_shape_matches_image(first_image, mask_dict[mask_selection]['raw']):
                new_quant = quantify_multiple_channels_per_roi(image_dict, mask_dict[mask_selection]['raw'],
                            data_selection, channels_to_quantify, aliases, dataset_options, delimiter, mask_selection)
                quant_frame = concat_quantification_frames_multi_roi(pd.DataFrame(cur_quant_dict), new_quant, data_selection, delimiter)
                return SessionServerside(quant_frame.to_dict(orient="records"), key=f"quantification_dict_{sesh_id}",
                        use_unique_key=app_config['serverside_overwrite']), dash.no_update, {'display': 'None'}, None
            else:
                error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["invalid_dimensions"])
                return dash.no_update, error_config, dash.no_update, dash.no_update
        else:
            error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["quantification_missing"])
            return dash.no_update, error_config, dash.no_update, dash.no_update

    @dash_app.callback(
        Output('stitch-preview-modal', 'is_open'),
        Output('stitch-preview-row', 'children'),
        Input('stitch-image-preview', "n_clicks"),
        State('stitch-image-select', 'value'),
        State('stitched_images', 'data'),
        prevent_initial_call=True)
    def preview_stitched_image(preview_stitch, stitch_select, stitch_collection):
        """
        Open the modal to preview the currently selected stitch image
        """
        return stitch_image_preview(stitch_collection, stitch_select)

    @dash_app.callback(
        Output('stitched_images', 'data', allow_duplicate=True),
        Output('stitch-image-select', 'options'),
        Input('stitch-image-create', "n_clicks"),
        State('session_id_internal', 'data'),
        State('stitch-image-create-width', 'value'),
        State('stitch-image-create-height', 'value'),
        State('stitch-image-id', 'value'),
        State('stitched_images', 'data'),
        Input('stitch-image-delete', 'n_clicks'),
        State('stitch-image-select', 'value'),
        prevent_initial_call=True)
    def modify_stitched_images(create_stitch, sesh_id, width, height, stitch_name, cur_stitch, delete_stitch, stitch_select):
        """
        Create or delete stitched images by name, saved to the server side cache
        """
        # set the name if the stitch is to be added or deleted
        stitch_name = stitch_select if ctx.triggered_id == 'stitch-image-delete' else stitch_name
        if sesh_id and stitch_name and (create_stitch or delete_stitch):
            cur_stitch = modify_stitch_cache(cur_stitch, stitch_name, height, width, stitch_cache_delete(ctx.triggered_id, stitch_name))
            return SessionServerside(cur_stitch, key=f"stitch_cache_{sesh_id}",
                        use_unique_key=app_config['serverside_overwrite']), stitch_cache_dropdown_labels(cur_stitch)
        raise PreventUpdate

    @dash_app.callback(
        Output("download-stitch-image", "data"),
        Input("stitch-image-download", "n_clicks"),
        State('stitched_images', 'data'),
        State('stitch-image-select', 'value'))
    @DownloadDirGenerator(os.path.join(tmpdirname, authentic_id))
    def download_switch_image(download_stitch, stitch_cache, stitch_select):
        """
        Download a stitched image in zip format due to potential size
        """
        if None not in (download_stitch, stitch_cache, stitch_select) and stitch_select in stitch_cache:
            download_stitch = os.path.join(str(download_stitch), str(uuid.uuid1()), 'downloads', 'stitch')
            return dcc.send_file(download_stitch_image(download_stitch, stitch_cache, stitch_select))
        raise PreventUpdate

    @dash_app.callback(
        Output("stitch-image-create-width", "value"),
        Output("stitch-image-create-height", "value"),
        Input("cur-roi-slide-parse", "n_clicks"),
        State('data-collection', 'value'),
        State('dataset-delimiter', 'value'),
        State('session_config', 'data'))
    def parse_roi_slide_for_stitch(parse_for_slide, roi_selection, delim, session_uploads):
        """
        Get the desired width and height of a slide image from the current ROI, if from MCD or CosMX Anndata
        """
        if None not in (roi_selection, delim, session_uploads):
            if roi_from_anndata_file(session_uploads, roi_selection, delim):
                return cosmx_global_slide_boundaries(roi_from_anndata_file(session_uploads, roi_selection, delim))
            return MCDAcqCoordinateParser(session_uploads, roi_selection, delim).get_roi_slide_boundary_point()
        raise PreventUpdate

    @dash_app.callback(
        Output("stitch-image-x-min", "value"),
        Output("stitch-image-y-min", "value"),
        Input("cur-roi-stitch-parse", "n_clicks"),
        Input('data-collection', 'value'),
        State('dataset-delimiter', 'value'),
        State('session_config', 'data'))
    def parse_roi_for_global_slide_coords(parse_for_slide, roi_selection, delim, session_uploads):
        """
        Get the desired x and y min in the global slide coordinate system for the current ROI, if from MCD
        """
        # if the ROI is switched, by default make the coordinates blank
        if ctx.triggered_id == "data-collection": return None, None
        if None not in (roi_selection, delim, session_uploads):
            if roi_from_anndata_file(session_uploads, roi_selection, delim):
                return cosmx_local_fov_position(roi_from_anndata_file(session_uploads, roi_selection, delim))
            return MCDAcqCoordinateParser(session_uploads, roi_selection, delim).get_roi_coord_min()
        raise PreventUpdate

    @dash_app.callback(
        Output("stitch-size-alert-modal", "is_open"),
        Output("stitch-size-information", "children"),
        Input('stitch-image-create-width', 'value'),
        Input('stitch-image-create-height', 'value'),
        State('toggle-session-messages', 'value'),
        prevent_initial_call=True)
    def check_for_large_stitch_dimensions(stitch_width, stitch_height, show_messages):
        """
        Check if any of the input stitch dimensions are large enough to warrant an alert (any dimension > 50000 pixels)
        """
        if show_messages and None not in (stitch_height, stitch_width) and any(int(dim) > 50000 for dim in (stitch_height, stitch_width)):
            return True, [html.H6("Message: \n"), html.H6(AlertMessage().warnings["large-stitch-dim"])]
        return False, None
