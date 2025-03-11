"""Module containing tools for outputting ROI annotations in different formats
such as PDF, tiff (zipped), or CSV
"""

import os
import json
from typing import Union
import shutil
import ast
import tifffile
import numpy as np
from dash import dcc
import pandas as pd
from dash.exceptions import PreventUpdate
from rakaia.inputs.pixel import set_roi_identifier_from_length
from rakaia.io.session import create_download_dir
from rakaia.utils.object import (
    get_min_max_values_from_zoom_box,
    get_min_max_values_from_rect_box,
    validate_mask_shape_matches_image,
    get_objs_in_svg_boundary_by_mask_percentage)
from rakaia.utils.pixel import get_region_dim_from_roi_dictionary, split_string_at_pattern
from rakaia.utils.pixel import path_to_mask

class AnnotationRegionWriter:
    """
    Provides an export of annotations that are not points to a CSV file.
    Export should be done to retrieve the object ids (i.e. cell ids) from the masks for each annotation
    This export can effectively replace the export of a quantification sheet with matched annotations in the event
    that quantification results are not available

    :param annotation_dict: Dictionary of annotations for the current ROI
    :param data_selection: String representation of the current session ROI
    :param mask_dict: Dictionary of the imported mask options
    :param delimiter: String expression on which to split the `data_selection` parameter into the filename
    :param use_roi_name: Whether to use the ROI name identifier in the column output
    :return: None
    """
    def __init__(self, annotation_dict: dict, data_selection: str, mask_dict: dict, delimiter: str="+++",
                 use_roi_name=True):
        self.annotation_dict = annotation_dict
        self.roi_selection = data_selection
        exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
        self.acquisition_name = acq
        self.mask_dict = mask_dict
        self.region_object_frame = {"ROI": [], "mask_name": [], "cell_id": [], "annotation_col": [], "annotation": []}
        self.filepath = None
        self.out_name = set_roi_identifier_from_length(self.roi_selection, delimiter=delimiter) if \
            use_roi_name else None

    def set_objects_included(self, key: Union[str, tuple], value: dict):
        """
        Set the list of objects to be included in the region writer based on the annotation type

        :param key: The unique identifier for the region annotation
        :param value: The dictionary containing the region annotation configuration information
        :return: List of included objects for a specific annotation
        """
        objects_included = []
        if value['type'] == 'path' and value['mask_selection'] is not None:
            objects_included = get_objs_in_svg_boundary_by_mask_percentage(
                mask_array=self.mask_dict[value['mask_selection']]["raw"], svgpath=key).keys()
        elif value['type'] == 'gate' and value['mask_selection'] is not None:
            objects_included = list(key)
        return objects_included

    def write_csv(self, dest_dir: str, dest_file: str="region_annotations.csv"):
        """
        Write the mask objects associated with regions to a CSV file

        :param dest_dir: Output directory for the CSV
        :param dest_file: Filename output for the CSV
        :return: `pd.DataFrame` of mask objects per region
        """
        create_download_dir(dest_dir)
        dest_file = f"{self.out_name}_regions.csv" if self.out_name else dest_file
        self.filepath = str(os.path.join(dest_dir, dest_file))
        if self.roi_selection in self.annotation_dict.keys() and \
                len(self.annotation_dict[self.roi_selection].items()) > 0:
            for key, value in self.annotation_dict[self.roi_selection].items():
                objects_included = self.set_objects_included(key, value)
                if objects_included:
                    for obj in objects_included:
                        if obj:
                            self.region_object_frame['ROI'].append(self.acquisition_name)
                            self.region_object_frame['mask_name'].append(str(value['mask_selection']))
                            self.region_object_frame['cell_id'].append(int(obj))
                            self.region_object_frame['annotation_col'].append(str(value['annotation_column']))
                            self.region_object_frame['annotation'].append(str(value['cell_type']))
        if not pd.DataFrame(self.region_object_frame).empty:
            pd.DataFrame(self.region_object_frame).to_csv(self.filepath, index=False)
            return self.filepath
        return None

class AnnotationMaskWriter:
    """
    Writes a series of region annotations to mask as tiff arrays

    :param annotation_dict: Dictionary of annotations for the current ROI
    :param data_selection: String representation of the current session ROI
    :param mask_shape: Tuple of the current mask dimensions
    :param canvas_mask: numpy array of the current mask applied to the canvas
    :param auto_create_dir: Whether to generate the output directory on initialization. Default is True
    :return: None
    """
    def __init__(self, dest_dir: str=None, annotation_dict: dict=None, data_selection: str=None,
                 mask_shape: tuple=None, canvas_mask: np.ndarray=None, auto_create_dir: bool=True):
        self.annotation_dict = annotation_dict
        self.dest_dir = os.path.join(dest_dir, "annotation_masks") if dest_dir is not None else None
        self.data_selection = data_selection
        self.mask_shape = mask_shape
        self.mask = canvas_mask
        self.cell_class_arrays = {}
        self.cell_class_ids = {}
        self.auto_create_dir = auto_create_dir

    def create_dir(self):
        """
        Create the directory to write the mask zip file

        :return: None
        """
        if os.path.exists(self.dest_dir) and os.access(os.path.dirname(self.dest_dir), os.R_OK) and \
                self.auto_create_dir:
            shutil.rmtree(os.path.dirname(self.dest_dir), ignore_errors=True)
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def add_new_category_mask(self, value):
        """
        Add a new empty annotation mask to the object array

        :return: None
        """
        if value['annotation_column'] not in self.cell_class_arrays:
            self.cell_class_arrays[value['annotation_column']] = np.zeros(self.mask_shape)

    def add_new_cell_type_category(self, value):
        """
        Add a new annotation category to the object array

        :return: None
        """
        if value['annotation_column'] not in self.cell_class_ids:
            self.cell_class_ids[value['annotation_column']] = {"counts": 0, 'annotations': {}}

    def add_new_count_type_id_match(self, value):
        """
        Increment the object ID counter for a specific annotation category

        :param value: The annotation associated with a specific annotation category
        :return: None
        """
        if value['cell_type'] not in self.cell_class_ids[value['annotation_column']]['annotations']:
            self.cell_class_ids[value['annotation_column']]['counts'] += 1
            self.cell_class_ids[value['annotation_column']]['annotations'][value['cell_type']] = \
                self.cell_class_ids[value['annotation_column']]['counts']

    @staticmethod
    def set_rectangular_region_bounds(key: Union[str, tuple], value: dict):
        """
        Set the min and max coordinate bounds for a rectangular region from either a canvas scroll zoom
        or a drawn rectangle

        :param key: The unique identifier for the region annotation
        :param value: The dictionary containing the region annotation configuration information
        :return: Tuple of x and y min and max coordinates
        """
        x_min, x_max, y_min, y_max = None, None, None, None
        if value['type'] == "zoom":
            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
        elif value['type'] == "rect":
            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))
        return x_min, x_max, y_min, y_max

    def write_annotation_masks(self):
        """
        Write the annotations to masks in a zip file and return the path of the zip

        :return: The zip filename containing all the zipped mask tiffs and JSON files
        """
        self.create_dir()
        for key, value in self.annotation_dict[self.data_selection].items():
            self.add_new_category_mask(value)
            # keep track of how many cell types are in each class
            self.add_new_cell_type_category(value)
            # if the cell type isn't there, add to the count and use the count as the mask ID
            self.add_new_count_type_id_match(value)
            self.cell_class_ids[value['annotation_column']]['mask_used'] = value['mask_selection']
            if value['type'] in ['point', 'points', 'path', 'zoom', 'rect']:
                # TODO: add mask output for cell gating
                if value['type'] not in ['point', 'points', 'path']:
                    x_min, x_max, y_min, y_max = self.set_rectangular_region_bounds(key, value)
                    # replace the array elements in the box with the count value
                    self.cell_class_arrays[value['annotation_column']][int(y_min):int(y_max), int(x_min):int(x_max)] = \
                        self.cell_class_ids[value['annotation_column']]['counts']
                    self.cell_class_arrays[value['annotation_column']] = self.cell_class_arrays[
                        value['annotation_column']].reshape(self.mask_shape)
                # if using an svg path, get the mask for the interior pixels
                elif value['type'] == 'point':
                    try:
                        x_coord = ast.literal_eval(key)['points'][0]['x']
                        y_coord = ast.literal_eval(key)['points'][0]['y']
                        if self.mask is not None:
                            cell_id = self.mask[y_coord, x_coord]

                            new_mask = np.where(self.mask == cell_id,
                                                self.cell_class_ids[value['annotation_column']]['counts'],
                                                self.cell_class_arrays[value['annotation_column']])
                            self.cell_class_arrays[value['annotation_column']] = new_mask
                        else:
                            self.cell_class_arrays[value['annotation_column']][y_coord, x_coord] = self.cell_class_ids[
                                value['annotation_column']]['counts']
                    except (KeyError, ValueError): pass
                elif value['type'] == 'path':
                    mask = path_to_mask(key, self.mask_shape)
                    self.cell_class_arrays[value['annotation_column']][mask] = \
                        self.cell_class_ids[value['annotation_column']]['counts']

        for name in self.cell_class_arrays.keys():
            original_name = f"{self.cell_class_ids[name]['mask_used']}_" if \
                self.cell_class_ids[name]['mask_used'] else ""
            mask_name = f"{original_name}{name}"
            tifffile.imwrite(os.path.join(self.dest_dir, f"{mask_name}.tiff"),
                             self.cell_class_arrays[name].astype(np.float32))
            with open(os.path.join(self.dest_dir, f"{mask_name}.json"), "w") as outfile:
                json.dump(self.cell_class_ids[name], outfile)
        shutil.make_archive(self.dest_dir, 'zip', self.dest_dir)
        return str(self.dest_dir + ".zip")

def export_point_annotations_as_csv(n_clicks, roi_name, annotations_dict, data_selection,
                                    mask_dict, apply_mask, mask_selection, image_dict,
                                    authentic_id, tmpdirname, delimiter: str="+++", use_roi_name: bool=True):
    """
    Parse through the dictionary of annotations and export the point annotations to a CSV file
    """
    if n_clicks > 0 and None not in (annotations_dict, data_selection):
        points = {'ROI': [], 'x': [], 'y': [], 'annotation_col': [], 'annotation': []}
        try:
            if None not in (image_dict, data_selection):
                first_image = get_region_dim_from_roi_dictionary(image_dict[data_selection])
            else:
                first_image = None
        except (KeyError, TypeError): first_image = None
        dest_path = os.path.join(tmpdirname, authentic_id, 'downloads', 'annotation_masks')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        # check that the mask is compatible with the current image
        if None not in (mask_dict, mask_selection) and apply_mask and first_image is not None and \
                validate_mask_shape_matches_image(first_image, mask_dict[mask_selection]['raw']):
            mask_used = mask_dict[mask_selection]['raw']
            points[mask_selection + "_cell_id"] = []
        else:
            mask_used = None
        for key, value in annotations_dict[data_selection].items():
            if value['type'] in ['point', 'points']:
                try:
                    points['x'].append(ast.literal_eval(key)['points'][0]['x'])
                    points['y'].append(ast.literal_eval(key)['points'][0]['y'])
                    points['annotation_col'].append(value['annotation_column'])
                    points['annotation'].append(value['cell_type'])
                    points['ROI'].append(roi_name)
                    if mask_used is not None:
                        cell_id = mask_used[ast.literal_eval(key)['points'][0]['y'],
                                ast.literal_eval(key)['points'][0]['x']].astype(int)
                        points[mask_selection + "_cell_id"].append(cell_id)
                except KeyError: pass
        if len(points['x']) > 0:
            frame = pd.DataFrame(points)
            out_name = set_roi_identifier_from_length(roi_name, delimiter=delimiter) if use_roi_name else ""
            return dcc.send_data_frame(frame.to_csv, f"{out_name}_points.csv", index=False)
        raise PreventUpdate
    raise PreventUpdate
