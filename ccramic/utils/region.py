import math
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path
from pydantic import BaseModel
from typing import Union
class ChannelRegion:
    """
    This abstract class defines a region for a particular channel
    """
    def __init__(self, channel_array, coordinates):
        self.channel_array = channel_array
        self.coordinate_dict = coordinates
        self.mean_exp = None
        self.max_exp = None
        self.min_exp = None
        # integrated has the sum of the signal in a region
        self.integrated = None

    def compute_pixel_mean(self):
        """
        Compute the mean pixel intensity of the channel region
        """
        return self.mean_exp

    def compute_pixel_min(self):
        """
        Compute the min pixel intensity of the channel region
        """
        return self.min_exp


    def compute_pixel_max(self):
        """
        Compute the max pixel intensity of the channel region
        """
        return self.max_exp

    def compute_integrated_signal(self):
        return self.integrated

class RectangularKeys(BaseModel):
    """
    Defines the possible keys for different rectangular regions on the canvas
    Options vary depending on if zoom is used, or a rectangular shape is drawn fresh
    or edited
    """
    keys: dict = {"zoom": ('xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[1]', 'yaxis.range[0]'),
                  "rect": ('x0', 'x1', 'y0', 'y1'),
                  "rect_redrawn": ('shapes[1].x0', 'shapes[1].x1', 'shapes[1].y0', 'shapes[1].y1')}

class RectangleRegion(ChannelRegion):
    """
    This class defines a channel region created using the zoom feature
    """
    def __init__(self, channel_array, coordinates, reg_type="zoom", redrawn=False):
        super().__init__(channel_array, coordinates)
        self.type = reg_type
        self.redrawn = redrawn
        self.key_dict = RectangularKeys().keys
        if self.type == "rect" and self.redrawn:
            self.required_keys = self.key_dict["rect_redrawn"]
        else:
            self.required_keys = self.key_dict[self.type]
        if all([elem in self.coordinate_dict] for elem in self.required_keys):
            try:
                if not all([elem >= 0 for elem in self.coordinate_dict.keys() if isinstance(elem, float)]):
                    raise AssertionError
                x_range_low = math.ceil(int(self.coordinate_dict[self.required_keys[0]]))
                x_range_high = math.ceil(int(self.coordinate_dict[self.required_keys[1]]))
                y_range_low = math.ceil(int(self.coordinate_dict[self.required_keys[2]]))
                y_range_high = math.ceil(int(self.coordinate_dict[self.required_keys[3]]))
                if not x_range_high >= x_range_low: raise AssertionError
                if not y_range_high >= y_range_low: raise AssertionError

                self.mean_exp, self.max_exp, self.min_exp, self.integrated = get_area_statistics_from_rect(
                    self.channel_array, x_range_low, x_range_high, y_range_low, y_range_high)
            except KeyError:
                self.mean_exp, self.max_exp, self.min_exp, self.integrated = 0, 0, 0, 0

class FreeFormRegion(ChannelRegion):
    """
    This class defines a channel region created by drawing a freeform SVG path shape
    """

    def __init__(self, channel_array, coordinates):
        super().__init__(channel_array, coordinates)
        if 'path' in self.coordinate_dict:
            self.path = self.coordinate_dict['path']
        # elif isinstance(self.coordinate_dict, dict) and \
        #         all(['shapes' in elem and 'path' in elem for elem in self.coordinate_dict.keys()]):
        #     self.path = list(self.coordinate_dict.values())[0]
        elif isinstance(self.coordinate_dict, str):
            self.path = self.coordinate_dict
        else:
            self.path = None
        if self.path is not None:
            self.mean_exp, self.max_exp, self.min_exp, self.integrated = get_area_statistics_from_closed_path(
                self.channel_array, self.path)
        else:
            self.mean_exp, self.max_exp, self.min_exp, self.integrated = 0, 0, 0, 0

class RegionAnnotation(BaseModel):
    title: str = None
    body: str = None
    cell_type: str = None
    imported: bool = False
    annotation_column: str = 'ccramic_cell_annotation'
    type: str = None
    channels: list = []
    use_mask: Union[bool, list, str] = None
    mask_selection: str = None
    mask_blending_level: float = 35.0
    add_mask_boundary: Union[bool, list, str] = True

def check_for_valid_annotation_hash(annotations_dict: dict=None, roi_selection: str=None):
    """
    Check the current annotation hash table, and create a new one if id doesn't exist,
    or add in the current ROI if not present
    """
    if annotations_dict is None or len(annotations_dict) < 1:
        annotations_dict = {}
    if roi_selection and roi_selection not in annotations_dict.keys():
        annotations_dict[roi_selection] = {}
    return annotations_dict
