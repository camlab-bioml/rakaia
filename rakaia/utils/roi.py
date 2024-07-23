import pandas as pd
import numpy as np
from PIL import Image
from rakaia.utils.object import validate_mask_shape_matches_image

def generate_dict_of_roi_cell_ids(measurements, sample_col="description", cell_id_col="cell_id"):
    """
    Generate a dictionary where each key is an ROI name from the query, and each value is a list of cell ids
    Used for subsetting the mask to display in the ROI gallery to indicate where cells are in the overall image
    """
    # use description as the default sample column, otherwise use sample
    measurements = pd.DataFrame(measurements)
    sample_col = sample_col if sample_col in measurements.columns else "sample"
    if sample_col not in measurements.columns or cell_id_col not in measurements.columns:
        return None
    else:
        cell_id_dict = {}
        for sample in list(measurements[sample_col].unique()):
            cell_id_dict[sample] = list(measurements[measurements[sample_col] == sample][cell_id_col].unique())
        return cell_id_dict


def subset_mask_outline_using_cell_id_list(mask_outline, original_mask, cell_id_list):
    """
    Subset a mask outline array to retain the cell outlines corresponding only to cell ids in the provided list
    Requires both the outline and the original mask as the outlines mask doesn't retain cell ids after the transformation
    """
    if len(original_mask.shape) > 2:
        original_mask = original_mask[:, :, 0]
    if not validate_mask_shape_matches_image(original_mask, mask_outline): return None
    mask_bool = np.isin(original_mask, cell_id_list)
    mask_outline[~mask_bool] = 0
    # converted = (mask_outline * 255).clip(0, 255).astype(np.uint8)
    return np.array(Image.fromarray(mask_outline.astype(np.float32)).convert('RGB')).astype(np.uint8)

def override_roi_gallery_blend_list(currently_selected: list, saved_blend_dict: dict=None,
                                    saved_blend:str=None):
    """
    Override the roi gallery blend list (channels) if a saved blend is used
    """
    if saved_blend_dict and saved_blend and saved_blend in saved_blend_dict:
        return [i for i in saved_blend_dict[saved_blend]]
    return currently_selected
