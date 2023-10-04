import numpy as np
from ..utils.cell_level_utils import *
from ..utils.pixel_level_utils import *
import os
import tifffile
import json
import shutil
import dash_core_components as dcc
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


def export_point_annotations_as_csv(n_clicks, annotations_dict, data_selection,
                                    mask_dict, apply_mask, mask_selection, image_dict,
                                    authentic_id, tmpdirname):
    """
    Parse through the dictionary of annotations and export the point annotations to a CSV file
    """
    if n_clicks > 0 and None not in (annotations_dict, data_selection):
        points = {'x': [], 'y': [], 'annotation_col': [], 'annotation': []}
        first_image = list(image_dict[data_selection].keys())[0]
        first_image = image_dict[data_selection][first_image]
        dest_path = os.path.join(tmpdirname, authentic_id, 'downloads', 'annotation_masks')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        # check that the mask is compatible with the current image
        if None not in (mask_dict, mask_selection) and apply_mask and \
                validate_mask_shape_matches_image(first_image, mask_dict[mask_selection]['raw']):
            mask_used = mask_dict[mask_selection]['raw']
            points[mask_selection] = []
        else:
            mask_used = None
        for key, value in annotations_dict[data_selection].items():
            if value['type'] in ['point', 'points']:
                try:
                    points['x'].append(eval(key)['points'][0]['x'])
                    points['y'].append(eval(key)['points'][0]['y'])
                    points['annotation_col'].append(value['annotation_column'])
                    points['annotation'].append(value['cell_type'])
                    if mask_used is not None:
                        cell_id = mask_used[eval(key)['points'][0]['y'], eval(key)['points'][0]['x']].astype(int)
                        points[mask_selection].append(cell_id)
                except KeyError:
                    pass
        if len(points['x']) > 0:
            frame = pd.DataFrame(points)
            return dcc.send_data_frame(frame.to_csv, "point_annotations.csv", index=False)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate
