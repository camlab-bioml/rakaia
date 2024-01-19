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
    def __init__(self, annotation_dict: dict, data_selection: str, mask_dict: dict):
        self.annotation_dict = annotation_dict
        self.roi_selection = data_selection
        exp, slide, acq = split_string_at_pattern(data_selection)
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


def export_annotations_as_masks(annotation_dict, output_dir, data_selection, mask_shape, canvas_mask=None):
    """
    Export the annotations contained within a dictionary as mask tiffs
    Create a mask tiff for each of the cell type classes, aka classification columns
    Additionally, export a JSON linking the mask IDs to the cell types
    use the tmpdir to write files and create a zip
    Return the zip path for dash
    """
    cell_class_arrays = {}
    cell_class_ids = {}
    for key, value in annotation_dict[data_selection].items():
        if value['annotation_column'] not in cell_class_arrays:
            cell_class_arrays[value['annotation_column']] = np.zeros(mask_shape)
        # keep track of how many cell types are in each class
        if value['annotation_column'] not in cell_class_ids:
            cell_class_ids[value['annotation_column']] = {"counts": 0, 'annotations': {}}
        # if the cell type isn't there, add to the count and use the count as the mask ID
        if value['cell_type'] not in cell_class_ids[value['annotation_column']]['annotations']:
            cell_class_ids[value['annotation_column']]['counts'] += 1
            cell_class_ids[value['annotation_column']]['annotations'][value['cell_type']] = \
                cell_class_ids[value['annotation_column']]['counts']
        if value['type'] in ['point', 'points', 'path', 'zoom', 'rect']:
            if value['type'] not in ['point', 'points', 'path']:
                if value['type'] == "zoom":
                    x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
                elif value['type'] == "rect":
                    x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))

                # replace the array elements in the box with the count value
                cell_class_arrays[value['annotation_column']][int(y_min):int(y_max), int(x_min):int(x_max)] = \
                    cell_class_ids[value['annotation_column']]['counts']
                cell_class_arrays[value['annotation_column']] = cell_class_arrays[value['annotation_column']].reshape(mask_shape)
            # if using an svgpath, get the mask for the interior pixels
            elif value['type'] == 'point':
                try:
                    x = eval(key)['points'][0]['x']
                    y = eval(key)['points'][0]['y']
                    if canvas_mask is not None:
                        cell_id = canvas_mask[y, x]

                        new_mask = np.where(canvas_mask == cell_id,
                                        cell_class_ids[value['annotation_column']]['counts'],
                                        cell_class_arrays[value['annotation_column']])
                        cell_class_arrays[value['annotation_column']] = new_mask
                    else:
                        cell_class_arrays[value['annotation_column']][y, x] = cell_class_ids[
                            value['annotation_column']]['counts']
                except (KeyError, ValueError):
                    pass
            elif value['type'] == 'path':
                mask = path_to_mask(key, mask_shape)
                cell_class_arrays[value['annotation_column']][mask] = \
                cell_class_ids[value['annotation_column']]['counts']

    # create the tmpdir where the tiffs and CSV will be written
    dest_path = os.path.join(output_dir, "annotation_masks")
    if os.path.exists(dest_path) and os.access(os.path.dirname(dest_path), os.R_OK):
        shutil.rmtree(os.path.dirname(dest_path))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for name, array in cell_class_arrays.items():
        tifffile.imwrite(os.path.join(dest_path, f"{name}.tiff"), array.astype(np.float32))
    for name, labels in cell_class_ids.items():
        with open(os.path.join(dest_path, f"{name}.json"), "w") as outfile:
            json.dump(labels, outfile)
    # TODO: convert the hash tables linking IDs to the cell type and save to dir
    shutil.make_archive(dest_path, 'zip', dest_path)
    return str(dest_path + ".zip")


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
