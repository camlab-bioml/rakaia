"""Advanced functions that support the callbacks associated with objects
(masking, clustering, UMAP, annotation, etc.)"""

from functools import partial
from typing import Union
import ast
import dash
import pandas as pd
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from rakaia.io.session import SessionServerside
from rakaia.utils.object import (
    populate_object_annotation_column_from_bounding_box,
    get_objs_in_svg_boundary_by_mask_percentage,
    populate_obj_annotation_column_from_obj_id_list,
    populate_obj_annotation_column_from_clickpoint,
    remove_annotation_entry_by_indices)
from rakaia.utils.pixel import get_bounding_box_for_svg_path
from rakaia.components.canvas import CanvasLayout
from rakaia.utils.alert import AlertMessage, add_warning_to_error_config
from rakaia.utils.shapes import filter_annotation_shapes

def transfer_annotations_by_index(annotations: dict, current_roi: Union[str, None]=None,
                                  target_roi: Union[str, None]=None, indices_to_transfer: Union[list, int]=None) -> dict:
    """
    Transfer one or more annotation indices from a current ROI to a target ROI.
    """
    indices_to_transfer = [indices_to_transfer] if isinstance(indices_to_transfer, int) else indices_to_transfer
    if current_roi and target_roi and indices_to_transfer and current_roi in annotations and \
            current_roi != target_roi:
        key_list = list(annotations[current_roi].keys())
        if target_roi not in annotations:
            annotations[target_roi] = {}
        try:
            for index_add in indices_to_transfer:
                annotations[target_roi][key_list[index_add]] = annotations[current_roi][key_list[index_add]]
        except (KeyError, IndexError):
            pass
    return annotations


class AnnotationQuantificationMerge:
    """
    Iterate a dictionary of ROI annotations and align them to a pandas dataframe of quantification results

    :param annotations: Dictionary of annotations for the current ROI
    :param quantification_frame: `pd.DataFrame` of object quantification results
    :param data_selection: String representation of the current ROI selection
    :param mask_config: dictionary of imported mask arrays that are selectable
    :param mask_toggle: Whether a mask is currently being applied
    :param mask_selection: Mask from the `mask_config` currently applied
    :param sample_name: Current ROI identifier
    :param id_column: Column in the quantification frame linking the mask to the measured results. Default is `sample`.
    :param config: Dictionary of current rakaia session settings from CLI
    :param remove: Whether the annotation(s) should be removed. Default is False
    :param indices_remove: List of ROI indices from the preview table to be removed
    :return: None
    """
    def __init__(self, annotations, quantification_frame, data_selection,
                mask_config, mask_toggle, mask_selection, sample_name=None, id_column='sample',
                config: dict=None, remove: bool=False, indices_remove: list=None,
                session_id: str="") -> None:
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
        self.session_id = session_id
        self.path = partial(self.populate_quantification_from_svgpath)
        self.rect = partial(self.populate_quantification_from_rectangle)
        self.point = partial(self.populate_quantification_from_clickpoint)

        if self.annotations:
            self.remove_annotations()
            if self.data_selection in self.annotations.keys() and \
                    len(self.annotations[self.data_selection]) > 0 and \
                    self.truthy_quantification(self.quantification_frame):
                self.quantification_frame = pd.DataFrame(quantification_frame)
                self.process_annotations(self.quantification_frame)
            if self.remove:
                self.annotations = remove_annotation_entry_by_indices(self.annotations,
                                               self.data_selection, self.indices_remove)

    def process_annotations(self, quantification_frame: Union[dict, pd.DataFrame]):
        """
        Iterate through a series of annotations, identify the type and the objects in the quantification
        results, and merge the annotation into a designated annotation column in the quantification frame

        :param quantification_frame: Frame of mask objects with summarized channel expression

        :return: None
        """
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
                        objects_included = self.get_objects_included(annotation)
                        self.quantification_frame = populate_obj_annotation_column_from_obj_id_list(
                            self.quantification_frame, obj_list=list(objects_included.keys()),
                            obj_type=self.annotations[self.data_selection][annotation]['cell_type'],
                            sample_name=self.sample_name,
                            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                            id_column=self.identifier, remove=self.remove)
                    else:
                        # use partial functions for the rest of the possibilities
                        self.quantification_frame = getattr(self,
                            self.annotations[self.data_selection][annotation]['type'])(annotation)
                    self.annotations[self.data_selection][annotation]['imported'] = True

    @staticmethod
    def truthy_quantification(quantification_results: Union[dict, pd.DataFrame]) -> bool:
        """
        Return if the current quantification results are truthy

        :param quantification_results: `pd.DataFrame` of object quantification results
        :return: boolean for truthy quantification results
        """
        if isinstance(quantification_results, pd.DataFrame):
            return not quantification_results.empty
        return bool(quantification_results)

    def remove_annotations(self) -> None:
        """
        Remove annotations from the annotation hash

        :return: None
        """
        if self.remove and self.data_selection in self.annotations and self.annotations[self.data_selection]:
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

    def get_objects_included(self, annotation) -> dict:
        """
        Get the objects associated with an annotation from a mask

        :param annotation: The ROI annotation dict
        :return: dictionary of objects included with the proportion of overlap for each ID
        """
        objects_included = {}
        # option 1: list from gated cells
        if self.annotations[self.data_selection][annotation]['type'] == "gate":
            objects_included = {obj: 100.0 for obj in list(annotation)}
        # if a mask is enabled, use the mask ID threshold method
        # otherwise, make a convex envelope bounding box
        # option 2: mask ID threshold from path
        elif self.annotations[self.data_selection][annotation]['type'] == "path":
            objects_included = get_objs_in_svg_boundary_by_mask_percentage(
                mask_array=self.mask_config[self.mask_selection]["raw"], svgpath=annotation)
        return objects_included

    def populate_quantification_from_zoom(self, annotation):
        """
        Populate the quantification frame annotation category from a zoom window annotation

        :param annotation: The ROI annotation dict
        :return: `pd.DataFrame` of quantification results with the annotated objects in the annotation category column
        """
        return populate_object_annotation_column_from_bounding_box(self.quantification_frame,
                            values_dict=dict(annotation),
                            obj_type=self.annotations[self.data_selection][annotation]['cell_type'],
                            annotation_column= self.annotations[self.data_selection][annotation]['annotation_column'],
                            remove=self.remove)

    def populate_quantification_from_svgpath(self, annotation):
        """
        Populate the quantification frame annotation category from a svg-path annotation drawn around a mask

        :param annotation: The ROI annotation dict
        :return: `pd.DataFrame` of quantification results with the annotated objects in the annotation category column
        """
        x_min, x_max, y_min, y_max = get_bounding_box_for_svg_path(annotation)
        val_dict = {'xaxis.range[0]': x_min, 'xaxis.range[1]': x_max,
                    'yaxis.range[0]': y_max, 'yaxis.range[1]': y_min}
        return populate_object_annotation_column_from_bounding_box(
            self.quantification_frame, values_dict=val_dict,
            obj_type=self.annotations[self.data_selection][annotation]['cell_type'],
            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
            remove=self.remove)

    def populate_quantification_from_rectangle(self, annotation):
        """
        Populate the quantification frame annotation category from a drawn rectangle

        :param annotation: The ROI annotation dict
        :return: `pd.DataFrame` of quantification results with the annotated objects in the annotation category column
        """
        return populate_object_annotation_column_from_bounding_box(
                            self.quantification_frame, values_dict=dict(annotation),
                            obj_type=self.annotations[self.data_selection][annotation]['cell_type'], box_type="rect",
                            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                            remove=self.remove)

    def populate_quantification_from_clickpoint(self, annotation):
        """
        Populate the quantification frame annotation category from a click point coordinate set

        :param annotation: The ROI annotation dict
        :return: `pd.DataFrame` of quantification results with the annotated objects in the annotation category column
        """
        return populate_obj_annotation_column_from_clickpoint(
                            self.quantification_frame, values_dict=ast.literal_eval(annotation),
                            obj_type=self.annotations[self.data_selection][annotation]['cell_type'],
                            annotation_column=self.annotations[self.data_selection][annotation]['annotation_column'],
                            mask_toggle=self.mask_toggle, mask_dict=self.mask_config,
                            mask_selection=self.mask_selection,
                            sample=self.sample_name, id_column=self.identifier, remove=self.remove)

    def get_annotated_frame(self) -> Union[dict, pd.DataFrame, None]:
        """
        Get the quantification result frame with the annotations added

        :return: `pd.DataFrame` of quantification results with the annotated objects in the annotation category column
        """
        return pd.DataFrame(self.quantification_frame).to_dict(orient="records") if \
            self.truthy_quantification(self.quantification_frame) else None

    def get_annotation_cache(self) -> SessionServerside:
        """
        Get the annotation hash in pickle transform format

        :return: `SessionServerside` transform of the annotation dictionary for all ROIs
        """
        return SessionServerside(self.annotations,
            key=f"annotation_dict_{self.session_id}", use_unique_key=self.config['serverside_overwrite'])

    def get_callback_structures(self) -> tuple:
        """
        Get the annotation hash in pickle transform format and quantification result frame with the annotations added

        :return: tuple: `pd.DataFrame` of quantification with annotations, and `annotation ROI SessionServerside`
        """
        if self.annotations:
            return self.get_annotated_frame(), self.get_annotation_cache()
        raise PreventUpdate

def callback_remove_canvas_annotation_shapes(n_clicks, cur_canvas, canvas_layout, error_config):
    """
    Remove any annotation shape on the canvas (i.e. any shape that is a rectangle or closed form svg path)
    """
    if n_clicks > 0 and None not in (cur_canvas, canvas_layout) and 'shapes' not in canvas_layout:
        cur_canvas = CanvasLayout(cur_canvas).clear_improper_shapes()
        if 'layout' in cur_canvas and 'shapes' in cur_canvas['layout']:
            cur_canvas['layout']['shapes'] = filter_annotation_shapes(cur_canvas)
            # IMP: to avoid the phantom shape set by https://github.com/plotly/dash/issues/2741
            # set the uirevision status to something different from what it was, BUT must still be truthy
            try:
                cur_canvas['layout']['uirevision'] = True if cur_canvas['layout']['uirevision'] not in [True] else "clear"
            except (KeyError, IndexError): pass
            fig = go.Figure(cur_canvas)
            fig.update_layout(dragmode="zoom")
            return fig, dash.no_update
        return go.Figure(cur_canvas), dash.no_update
    elif 'shapes' in canvas_layout or ('layout' in cur_canvas and 'shapes' in cur_canvas['layout'] and len(
                cur_canvas['layout']['shapes']) > 0):
        error_config = add_warning_to_error_config(error_config, AlertMessage().warnings["invalid_annotation_shapes"])
        return dash.no_update, error_config
    raise PreventUpdate

def reset_annotation_import(annotation_dict: dict=None, roi_selection: str=None, app_config: dict=None,
                            return_as_serverside: bool=True, session_id: str=""):
    """
    Reset the import of an annotation from the ROI dictionary
    """
    if annotation_dict and roi_selection and roi_selection in annotation_dict:
        for value in annotation_dict[roi_selection].values():
            value['imported'] = False
    return SessionServerside(annotation_dict, key=f"annotation_dict_{session_id}",
                             use_unique_key=app_config['serverside_overwrite']) if \
        return_as_serverside else annotation_dict
