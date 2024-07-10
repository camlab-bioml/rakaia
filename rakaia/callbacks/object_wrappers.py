import dash
import pandas as pd
from rakaia.io.session import SessionServerside
from rakaia.utils.object import (
    populate_cell_annotation_column_from_bounding_box,
    get_cells_in_svg_boundary_by_mask_percentage,
    populate_cell_annotation_column_from_cell_id_list,
    populate_cell_annotation_column_from_clickpoint, remove_annotation_entry_by_indices)
from rakaia.utils.pixel import get_bounding_box_for_svgpath
import ast
from rakaia.components.canvas import CanvasLayout
from dash.exceptions import PreventUpdate
from rakaia.utils.alert import AlertMessage
from rakaia.utils.shapes import filter_annotation_shapes
import plotly.graph_objs as go

def callback_add_region_annotation_to_quantification_frame(annotations, quantification_frame, data_selection,
                                    mask_config, mask_toggle, mask_selection, sample_name=None, id_column='sample',
                                    config: dict=None, remove: bool=False, indices_remove: list=None):
    # loop through all of the existing annotations
    # for annotations that have not yet been imported, import and set the import status to True
    if annotations and len(annotations) > 0:
        # if removing the latest annotation, set it to not imported, replace with the default of None, then delete
        if remove and annotations[data_selection]:
            # set the ids for the indices to remove
            if not indices_remove:
                last = list(annotations[data_selection].keys())[-1]
                annotations[data_selection][last]['imported'] = False
            else:
                id_list = [elem['id'] for elem in annotations[data_selection].values() if 'id' in elem]
                for index_position in indices_remove:
                    try:
                        id_annot = id_list[index_position]
                        for annotation in annotations[data_selection].values():
                            if annotation['id'] == id_annot:
                                annotation['imported'] = False
                    except IndexError:
                        pass
        if data_selection in annotations.keys() and len(annotations[data_selection]) > 0 and quantification_frame:
            quantification_frame = pd.DataFrame(quantification_frame)
            for annotation in annotations[data_selection].keys():
                if not annotations[data_selection][annotation]['imported']:
                    # import only the new annotations that are rectangles (for now) and are not validated
                    if annotations[data_selection][annotation]['type'] == "zoom":
                        quantification_frame = populate_cell_annotation_column_from_bounding_box(quantification_frame,
                        values_dict=dict(annotation), cell_type=annotations[data_selection][annotation]['cell_type'],
                        annotation_column=annotations[data_selection][annotation]['annotation_column'], remove=remove)
                    elif annotations[data_selection][annotation]['type'] in ["path", "gate"] and mask_toggle and \
                                None not in (mask_config, mask_selection) and len(mask_config) > 0:
                        cells_included = []
                        # option 1: list from gated cells
                        if annotations[data_selection][annotation]['type'] == "gate":
                            cells_included = {cell: 100.0 for cell in list(annotation)}
                        # if a mask is enabled, use the mask ID threshold method
                        # otherwise, make a convex envelope bounding box
                        # option 2: mask ID threshold from path
                        elif annotations[data_selection][annotation]['type'] == "path":
                            cells_included = get_cells_in_svg_boundary_by_mask_percentage(
                                mask_array=mask_config[mask_selection]["raw"], svgpath=annotation)
                        quantification_frame = populate_cell_annotation_column_from_cell_id_list(
                            quantification_frame, cell_list=list(cells_included.keys()),
                            cell_type=annotations[data_selection][annotation]['cell_type'], sample_name=sample_name,
                            annotation_column=annotations[data_selection][annotation]['annotation_column'],
                            id_column=id_column, remove=remove)
                        # option 2: convex envelope bounding box from path
                    elif annotations[data_selection][annotation]['type'] == "path":
                        x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(annotation)
                        val_dict = {'xaxis.range[0]': x_min, 'xaxis.range[1]': x_max,
                                        'yaxis.range[0]': y_max, 'yaxis.range[1]': y_min}
                        quantification_frame = populate_cell_annotation_column_from_bounding_box(
                                quantification_frame, values_dict=val_dict,
                                cell_type=annotations[data_selection][annotation]['cell_type'],
                                annotation_column=annotations[data_selection][annotation]['annotation_column'],
                                remove=remove)
                    elif annotations[data_selection][annotation]['type'] == "rect":
                        quantification_frame = populate_cell_annotation_column_from_bounding_box(
                            quantification_frame, values_dict=dict(annotation),
                            cell_type=annotations[data_selection][annotation]['cell_type'], box_type="rect",
                            annotation_column=annotations[data_selection][annotation]['annotation_column'],
                            remove=remove)
                    elif annotations[data_selection][annotation]['type'] == "point":
                        quantification_frame = populate_cell_annotation_column_from_clickpoint(
                            quantification_frame, values_dict=ast.literal_eval(annotation),
                            cell_type=annotations[data_selection][annotation]['cell_type'],
                            annotation_column=annotations[data_selection][annotation]['annotation_column'],
                            mask_toggle=mask_toggle, mask_dict=mask_config, mask_selection=mask_selection,
                            sample=sample_name, id_column=id_column, remove=remove)
                    annotations[data_selection][annotation]['imported'] = True
            # if remove, remove the last annotation from the dictionary
            if remove:
                annotations = remove_annotation_entry_by_indices(annotations, data_selection, indices_remove)
            return quantification_frame.to_dict(orient="records"), \
                SessionServerside(annotations, key="annotation_dict", use_unique_key=config['serverside_overwrite'])
        elif annotations:
            if remove:
                annotations = remove_annotation_entry_by_indices(annotations, data_selection, indices_remove)
            return None, SessionServerside(annotations,
            key="annotation_dict", use_unique_key=config['serverside_overwrite'])
    else:
        raise PreventUpdate

def callback_remove_canvas_annotation_shapes(n_clicks, cur_canvas, canvas_layout, error_config):
    """
    Remove any annotation shape on the canvas (i.e. any shape that is a rectangle or closed form svgpath)
    """
    if n_clicks > 0 and None not in (cur_canvas, canvas_layout) and 'shapes' not in canvas_layout:
        cur_canvas = CanvasLayout(cur_canvas).clear_improper_shapes()
        if 'layout' in cur_canvas and 'shapes' in cur_canvas['layout']:
            cur_canvas['layout']['shapes'] = filter_annotation_shapes(cur_canvas)
            # IMP: to avoid the phantom shape set by https://github.com/plotly/dash/issues/2741
            # set the uirevision status to something different from what it was, BUT must still be truthy
            cur_canvas['layout']['uirevision'] = True if cur_canvas['layout']['uirevision'] not in [True] else "clear"
            fig = go.Figure(cur_canvas)
            fig.update_layout(dragmode="zoom")
            return fig, dash.no_update
        else:
            return go.Figure(cur_canvas), dash.no_update
    elif 'shapes' in canvas_layout or ('layout' in cur_canvas and \
                                       'shapes' in cur_canvas['layout'] and len(
                cur_canvas['layout']['shapes']) > 0):
        if error_config is None:
            error_config = {"error": None}
        error_config["error"] = AlertMessage().warnings["invalid_annotation_shapes"]
        return dash.no_update, error_config
    else:
        raise PreventUpdate

def reset_annotation_import(annotation_dict: dict=None, roi_selection: str=None, app_config: dict=None,
                            return_as_serverside: bool=True):
    if annotation_dict and roi_selection and roi_selection in annotation_dict:
        for value in annotation_dict[roi_selection].values():
            value['imported'] = False
    return SessionServerside(annotation_dict, key="annotation_dict",
                             use_unique_key=app_config['serverside_overwrite']) if \
        return_as_serverside else annotation_dict
