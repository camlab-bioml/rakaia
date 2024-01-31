import numpy as np
from ccramic.io.session import create_download_dir
from ccramic.utils.cell_level_utils import (
    get_min_max_values_from_zoom_box,
    get_min_max_values_from_rect_box,
    validate_mask_shape_matches_image,
    get_cells_in_svg_boundary_by_mask_percentage)
from ccramic.utils.pixel_level_utils import get_first_image_from_roi_dictionary, split_string_at_pattern
from ccramic.utils.pixel_level_utils import path_to_mask
import os
import tifffile
import json
import shutil
from dash import dcc
import pandas as pd
from dash.exceptions import PreventUpdate


class AnnotationRegionWriter:
    """
    Represents an export of annotations that are not points to a CSV file.
    Export should be done to retrieve the object ids (i.e. cell ids) from the masks for each annotation
    This export can effectively replace the export of a quantification sheet with matched annotations in the event
    that quantification results are not available
    """
    def __init__(self, annotation_dict: dict, data_selection: str, mask_dict: dict, delimiter: str="+++"):
        self.annotation_dict = annotation_dict
        self.roi_selection = data_selection
        exp, slide, acq = split_string_at_pattern(data_selection, pattern=delimiter)
        self.acquisition_name = acq
        self.mask_dict = mask_dict
        self.region_object_frame = {"ROI": [], "mask_name": [], "cell_id": [], "annotation_col": [], "annotation": []}
        self.filepath = None

    def write_csv(self, dest_dir: str, dest_file: str="region_annotations.csv"):
        create_download_dir(dest_dir)
        self.filepath = str(os.path.join(dest_dir, dest_file))
        if self.roi_selection in self.annotation_dict.keys() and \
            len(self.annotation_dict[self.roi_selection].items()) > 0:
            for key, value in self.annotation_dict[self.roi_selection].items():
                # TODO: for now, just use svg paths
                # make sure that the mask is not None so that the ids inside the mask can be extracted
                if value['type'] == 'path' and value['mask_selection'] is not None:
                    objects_included = get_cells_in_svg_boundary_by_mask_percentage(
                        mask_array=self.mask_dict[value['mask_selection']]["raw"], svgpath=key)
                    for obj in objects_included:
                        self.region_object_frame['ROI'].append(self.acquisition_name)
                        self.region_object_frame['mask_name'].append(str(value['mask_selection']))
                        self.region_object_frame['cell_id'].append(int(obj))
                        self.region_object_frame['annotation_col'].append(str(value['annotation_column']))
                        self.region_object_frame['annotation'].append(str(value['cell_type']))
        pd.DataFrame(self.region_object_frame).to_csv(self.filepath, index=False)
        return self.filepath

class AnnotationMaskWriter:
    """
    Writes a series of region annotations to mask as tiff arrays
    """
    def __init__(self, annotation_dict: dict, output_dir: str=None, data_selection: str=None,
                 mask_shape: tuple=None, canvas_mask: np.ndarray=None):
        self.annotation_dict = annotation_dict
        self.dest_dir = os.path.join(output_dir, "annotation_masks") if output_dir is not None else None
        self.data_selection = data_selection
        self.mask_shape = mask_shape
        self.mask = canvas_mask
        self.cell_class_arrays = {}
        self.cell_class_ids = {}

    def create_dir(self):
        """
        Create the directory to write the mask zip file
        """
        if os.path.exists(self.dest_dir) and os.access(os.path.dirname(self.dest_dir), os.R_OK):
            shutil.rmtree(os.path.dirname(self.dest_dir))
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def add_new_category_mask(self, value):
        if value['annotation_column'] not in self.cell_class_arrays:
            self.cell_class_arrays[value['annotation_column']] = np.zeros(self.mask_shape)

    def add_new_cell_type_category(self, value):
        if value['annotation_column'] not in self.cell_class_ids:
            self.cell_class_ids[value['annotation_column']] = {"counts": 0, 'annotations': {}}

    def add_new_count_type_id_match(self, value):
        if value['cell_type'] not in self.cell_class_ids[value['annotation_column']]['annotations']:
            self.cell_class_ids[value['annotation_column']]['counts'] += 1
            self.cell_class_ids[value['annotation_column']]['annotations'][value['cell_type']] = \
                self.cell_class_ids[value['annotation_column']]['counts']
    def write_annotation_masks(self):
        """
        Write the annotations to masks in a zip file and return the path of the zip
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
                if value['type'] not in ['point', 'points', 'path']:
                    if value['type'] == "zoom":
                        x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
                    elif value['type'] == "rect":
                        x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))

                    # replace the array elements in the box with the count value
                    self.cell_class_arrays[value['annotation_column']][int(y_min):int(y_max), int(x_min):int(x_max)] = \
                        self.cell_class_ids[value['annotation_column']]['counts']
                    self.cell_class_arrays[value['annotation_column']] = self.cell_class_arrays[
                        value['annotation_column']].reshape(self.mask_shape)
                # if using an svgpath, get the mask for the interior pixels
                elif value['type'] == 'point':
                    try:
                        x = eval(key)['points'][0]['x']
                        y = eval(key)['points'][0]['y']
                        if self.mask is not None:
                            cell_id = self.mask[y, x]

                            new_mask = np.where(self.mask == cell_id,
                                                self.cell_class_ids[value['annotation_column']]['counts'],
                                                self.cell_class_arrays[value['annotation_column']])
                            self.cell_class_arrays[value['annotation_column']] = new_mask
                        else:
                            self.cell_class_arrays[value['annotation_column']][y, x] = self.cell_class_ids[
                                value['annotation_column']]['counts']
                    except (KeyError, ValueError):
                        pass
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
        # for name, array in cell_class_arrays.items():
        #     # set the name of the mask to include the original mask name if provided, otherwise none
        #     original_name = f"{cell_class_ids[name]['name']}_" if cell_class_ids[name]['name'] else ""
        #     mask_name = f"{original_name}{name}"
        #     tifffile.imwrite(os.path.join(dest_path, f"{mask_name}.tiff"), array.astype(np.float32))
        # for name, labels in cell_class_ids.items():
        #     original_name = f"{cell_class_ids[name]['name']}_" if cell_class_ids[name]['name'] else ""
        #     mask_name = f"{original_name}{name}"
        #     with open(os.path.join(dest_path, f"{mask_name}.json"), "w") as outfile:
        #         json.dump(labels, outfile)
        # TODO: convert the hash tables linking IDs to the cell type and save to dir
        shutil.make_archive(self.dest_dir, 'zip', self.dest_dir)
        return str(self.dest_dir + ".zip")

def export_point_annotations_as_csv(n_clicks, roi_name, annotations_dict, data_selection,
                                    mask_dict, apply_mask, mask_selection, image_dict,
                                    authentic_id, tmpdirname):
    """
    Parse through the dictionary of annotations and export the point annotations to a CSV file
    """
    if n_clicks > 0 and None not in (annotations_dict, data_selection):
        points = {'ROI': [], 'x': [], 'y': [], 'annotation_col': [], 'annotation': []}
        try:
            if None not in (image_dict, data_selection):
                first_image = get_first_image_from_roi_dictionary(image_dict[data_selection])
            else:
                first_image = None
        except (KeyError, TypeError):
            first_image = None
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
                    points['x'].append(eval(key)['points'][0]['x'])
                    points['y'].append(eval(key)['points'][0]['y'])
                    points['annotation_col'].append(value['annotation_column'])
                    points['annotation'].append(value['cell_type'])
                    points['ROI'].append(roi_name)
                    if mask_used is not None:
                        cell_id = mask_used[eval(key)['points'][0]['y'], eval(key)['points'][0]['x']].astype(int)
                        points[mask_selection + "_cell_id"].append(cell_id)
                except KeyError:
                    pass
        if len(points['x']) > 0:
            frame = pd.DataFrame(points)
            return dcc.send_data_frame(frame.to_csv, "point_annotations.csv", index=False)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate
