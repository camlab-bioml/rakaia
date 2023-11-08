import pandas as pd
import numpy as np
from ccramic.utils.cell_level_utils import validate_mask_shape_matches_image

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
    try:
        original_reshape = original_mask.reshape((original_mask.shape[0], original_mask.shape[1]))
        assert validate_mask_shape_matches_image(original_reshape, mask_outline)
        mask_bool = np.isin(original_reshape, cell_id_list)
        mask_outline[~mask_bool] = 0
        return mask_outline
    except AssertionError:
        return None
