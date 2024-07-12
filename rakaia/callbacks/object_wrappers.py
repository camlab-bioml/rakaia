from functools import partial
from typing import Union
import ast
import dash
import pandas as pd
from rakaia.io.session import SessionServerside
from rakaia.utils.object import (
    populate_cell_annotation_column_from_bounding_box,
    get_cells_in_svg_boundary_by_mask_percentage,
    populate_cell_annotation_column_from_cell_id_list,
    populate_cell_annotation_column_from_clickpoint,
    remove_annotation_entry_by_indices)
from rakaia.utils.pixel import get_bounding_box_for_svgpath
from rakaia.components.canvas import CanvasLayout
from dash.exceptions import PreventUpdate
from rakaia.utils.alert import AlertMessage
from rakaia.utils.shapes import filter_annotation_shapes
import plotly.graph_objs as go

class AnnotationQuantificationMerge:
    """
    Iterate a dictionary of ROI annotations and align them to a pandas dataframe of quantification results
    """
    def __init__(self, annotations, quantification_frame, data_selection,
                mask_config, mask_toggle, mask_selection, sample_name=None, id_column='sample',
                config: dict=None, remove: bool=False, indices_remove: list=None):
        self.annotations = annotations
        self.quantification_frame = quantification_frame
        self.data_selection = data_selection
        self.mask_config = mask_config
        self.mask_toggle = mask_toggle
        self.mask_selection = mask_selection
        self.sample_name = sample_name
        self.identifier = id_column
        self.config = config
        self.remove = remove
        self.indices_remove = indices_remove
        self.path = partial(self.populate_quantification_from_svgpath)
        self.rect = partial(self.populate_quantification_from_rectangle)
        self.point = partial(self.populate_quantification_from_clickpoint)

        if self.annotations:
            self.remove_annotations()
            if self.data_selection in self.annotations.keys() and \
                    len(self.annotations[self.data_selection]) > 0 and \
                    self.truthy_quantification(self.quantification_frame):
                self.quantification_frame = pd.DataFrame(quantification_frame)
                for annotation in self.annotations[self.data_selection].keys():
                    if not self.annotations[self.data_selection][annotation]['imported']:
                        if self.annotations[self.data_selection][annotation]['type'] == "zoom":
                            self.quantification_frame = self.populate_quantification_from_zoom(annotation)
                        elif self.annotations[self.data_selection][annotation]['type'] in ["path", "gate"] and \
                                self.mask_toggle and None not in (self.mask_config, self.mask_selection) and \
                                len(self.mask_config) > 0:
                            cells_included = self.get_cells_included(annotation)
                            self.quantification_frame = populate_cell_annotation_column_from_cell_id_list(
                                self.quantification_frame, cell_list=list(cells_included.keys()),
                                cell_type=self.annotations[self.data_selection][annotation]['cell_type'],
                                sample_name=self.sample_name,
                                annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                                id_column=self.identifier, remove=self.remove)
                        else:
                            # use partial functions for the rest of the possibilities
                            self.quantification_frame = getattr(self,
                            self.annotations[self.data_selection][annotation]['type'])(annotation)
                        self.annotations[self.data_selection][annotation]['imported'] = True
            if self.remove:
                self.annotations = remove_annotation_entry_by_indices(self.annotations,
                                               self.data_selection, self.indices_remove)
    @staticmethod
    def truthy_quantification(quantification_results: Union[dict, pd.DataFrame]):
        if isinstance(quantification_results, pd.DataFrame):
            return not quantification_results.empty
        return bool(quantification_results)

    def remove_annotations(self):
        if self.remove and self.annotations[self.data_selection]:
            # set the ids for the indices to remove
            if not self.indices_remove:
                last = list(self.annotations[self.data_selection].keys())[-1]
                self.annotations[self.data_selection][last]['imported'] = False
            else:
                id_list = [elem['id'] for elem in self.annotations[self.data_selection].values() if 'id' in elem]
                for index_position in self.indices_remove:
                    try:
                        id_annot = id_list[index_position]
                        for annotation in self.annotations[self.data_selection].values():
                            if annotation['id'] == id_annot:
                                annotation['imported'] = False
                    except IndexError:
                        pass

    def get_cells_included(self, annotation):
        cells_included = []
        # option 1: list from gated cells
        if self.annotations[self.data_selection][annotation]['type'] == "gate":
            cells_included = {cell: 100.0 for cell in list(annotation)}
        # if a mask is enabled, use the mask ID threshold method
        # otherwise, make a convex envelope bounding box
        # option 2: mask ID threshold from path
        elif self.annotations[self.data_selection][annotation]['type'] == "path":
            cells_included = get_cells_in_svg_boundary_by_mask_percentage(
                mask_array=self.mask_config[self.mask_selection]["raw"], svgpath=annotation)
        return cells_included


    def populate_quantification_from_zoom(self, annotation):
        return populate_cell_annotation_column_from_bounding_box(self.quantification_frame,
                                    values_dict=dict(annotation),
                cell_type=self.annotations[self.data_selection][annotation]['cell_type'],
                annotation_column= self.annotations[self.data_selection][annotation]['annotation_column'],
                remove=self.remove)

    def populate_quantification_from_svgpath(self, annotation):
        x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(annotation)
        val_dict = {'xaxis.range[0]': x_min, 'xaxis.range[1]': x_max,
                    'yaxis.range[0]': y_max, 'yaxis.range[1]': y_min}
        return populate_cell_annotation_column_from_bounding_box(
            self.quantification_frame, values_dict=val_dict,
            cell_type=self.annotations[self.data_selection][annotation]['cell_type'],
            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
            remove=self.remove)

    def populate_quantification_from_rectangle(self, annotation):
        return populate_cell_annotation_column_from_bounding_box(
                            self.quantification_frame, values_dict=dict(annotation),
                            cell_type=self.annotations[self.data_selection][annotation]['cell_type'], box_type="rect",
                            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                            remove=self.remove)

    def populate_quantification_from_clickpoint(self, annotation):
        return populate_cell_annotation_column_from_clickpoint(
                            self.quantification_frame, values_dict=ast.literal_eval(annotation),
                            cell_type=self.annotations[self.data_selection][annotation]['cell_type'],
                            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                            mask_toggle=self.mask_toggle, mask_dict=self.mask_config,
                            mask_selection=self.mask_selection,
                            sample=self.sample_name, id_column=self.identifier, remove=self.remove)

    def get_annotated_frame(self):
        return pd.DataFrame(self.quantification_frame).to_dict(orient="records") if \
            self.truthy_quantification(self.quantification_frame) else None

    def get_annotation_cache(self):
        return SessionServerside(self.annotations,
            key="annotation_dict", use_unique_key=self.config['serverside_overwrite'])

    def get_callback_structures(self):
        if self.annotations:
            return self.get_annotated_frame(), self.get_annotation_cache()
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
