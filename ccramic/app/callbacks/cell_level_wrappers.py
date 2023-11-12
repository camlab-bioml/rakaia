import ast

import dash
import dash_uploader as du
import pandas as pd
from dash_extensions.enrich import Output, Input, State
from dash import ctx
from ..parsers.cell_level_parsers import *
from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *
from dash import dcc
import ast

def callback_add_region_annotation_to_quantification_frame(annotations, quantification_frame, data_selection,
                                                      mask_config, mask_toggle, mask_selection, sample_name=None,
                                                        id_column='sample'):
    # loop through all of the existing annotations
    # for annotations that have not yet been imported, import and set the import status to True
    if None not in (annotations, quantification_frame) and len(quantification_frame) > 0 and len(annotations) > 0:
        if data_selection in annotations.keys() and len(annotations[data_selection]) > 0:
            quantification_frame = pd.DataFrame(quantification_frame)
            for annotation in annotations[data_selection].keys():
                if not annotations[data_selection][annotation]['imported']:
                    # import only the new annotations that are rectangles (for now) and are not validated
                    if annotations[data_selection][annotation]['type'] == "zoom":
                        quantification_frame = populate_cell_annotation_column_from_bounding_box(quantification_frame,
                        values_dict=dict(annotation), cell_type=annotations[data_selection][annotation]['cell_type'],
                        annotation_column=annotations[data_selection][annotation]['annotation_column'])

                    elif annotations[data_selection][annotation]['type'] == "path":
                        # TODO: decide which method of annotation to use
                        # if a mask is enabled, use the mask ID threshold method
                        # otherwise, make a convex envelope bounding box

                        # option 1: mask ID threshold
                        if mask_toggle and None not in (mask_config, mask_selection) and len(mask_config) > 0:
                            cells_included = get_cells_in_svg_boundary_by_mask_percentage(
                                mask_array=mask_config[mask_selection]["raw"], svgpath=annotation)
                            quantification_frame = populate_cell_annotation_column_from_cell_id_list(
                                quantification_frame, cell_list=list(cells_included.keys()),
                                cell_type=annotations[data_selection][annotation]['cell_type'],
                            sample_name=sample_name,
                                annotation_column=annotations[data_selection][annotation]['annotation_column'],
                                id_column=id_column)
                        # option 2: convex envelope bounding box
                        else:
                            x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(annotation)
                            val_dict = {'xaxis.range[0]': x_min, 'xaxis.range[1]': x_max,
                                        'yaxis.range[0]': y_max, 'yaxis.range[1]': y_min}
                            quantification_frame = populate_cell_annotation_column_from_bounding_box(
                                quantification_frame, values_dict=val_dict,
                                cell_type=annotations[data_selection][annotation]['cell_type'],
                                annotation_column=annotations[data_selection][annotation]['annotation_column'])
                    elif annotations[data_selection][annotation]['type'] == "rect":
                        quantification_frame = populate_cell_annotation_column_from_bounding_box(
                            quantification_frame, values_dict=dict(annotation),
                            cell_type=annotations[data_selection][annotation]['cell_type'], box_type="rect",
                            annotation_column=annotations[data_selection][annotation]['annotation_column'])
                    elif annotations[data_selection][annotation]['type'] == "point":
                        quantification_frame = populate_cell_annotation_column_from_clickpoint(
                            quantification_frame, values_dict=ast.literal_eval(annotation),
                            cell_type=annotations[data_selection][annotation]['cell_type'],
                            annotation_column=annotations[data_selection][annotation]['annotation_column'],
                            mask_toggle=mask_toggle, mask_dict=mask_config, mask_selection=mask_selection,
                            sample=sample_name, id_column=id_column)
                    annotations[data_selection][annotation]['imported'] = True
            return quantification_frame.to_dict(orient="records"), Serverside(annotations)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

def callback_remove_canvas_annotation_shapes(n_clicks, cur_canvas, canvas_layout, error_config):
    """
    Remove any annotation shape on the canvas (i.e. any shape that is a rectangle or closed form svgpath)
    """
    if n_clicks > 0 and None not in (cur_canvas, canvas_layout) and 'shapes' not in canvas_layout:
        if 'layout' in cur_canvas and 'shapes' in cur_canvas['layout']:
            try:
                cur_canvas['layout']['shapes'] = [elem for elem in cur_canvas['layout']['shapes'] if \
                                                  'type' in elem and \
                                                  elem['type'] not in ['rect', 'path', 'circle']]
            except KeyError:
                pass
            return cur_canvas, dash.no_update
        else:
            raise PreventUpdate
    elif 'shapes' in canvas_layout or ('layout' in cur_canvas and \
                                       'shapes' in cur_canvas['layout'] and len(
                cur_canvas['layout']['shapes']) > 0):
        if error_config is None:
            error_config = {"error": None}
        error_config["error"] = "There are annotation shapes in the current layout. \n" \
                                "Switch to zoom or pan before removing the annotation shapes."
        return cur_canvas, error_config
    else:
        raise PreventUpdate
