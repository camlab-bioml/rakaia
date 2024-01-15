import math
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path
from pydantic import BaseModel

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

class RectangularKeys(BaseModel):
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
        # if self.type == "zoom":
        #     self.required_keys = ('xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[1]', 'yaxis.range[0]')
        # elif self.type == "rect":
        #     if self.redrawn:
        #         self.required_keys = ('shapes[1].x0', 'shapes[1].x1', 'shapes[1].y0', 'shapes[1].y1')
        #     else:
        #         self.required_keys = ('x0', 'x1', 'y0', 'y1')
        if all([elem in self.coordinate_dict] for elem in self.required_keys):
            try:
                assert all([elem >= 0 for elem in self.coordinate_dict.keys() if isinstance(elem, float)])
                x_range_low = math.ceil(int(self.coordinate_dict[self.required_keys[0]]))
                x_range_high = math.ceil(int(self.coordinate_dict[self.required_keys[1]]))
                y_range_low = math.ceil(int(self.coordinate_dict[self.required_keys[2]]))
                y_range_high = math.ceil(int(self.coordinate_dict[self.required_keys[3]]))
                assert x_range_high >= x_range_low
                assert y_range_high >= y_range_low

                self.mean_exp, self.max_exp, self.min_exp = get_area_statistics_from_rect(self.channel_array,
                                                                                          x_range_low,
                                                                                          x_range_high,
                                                                                          y_range_low, y_range_high)
            except (AssertionError, KeyError):
                self.mean_exp, self.max_exp, self.min_exp = 0, 0, 0

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
            self.mean_exp, self.max_exp, self.min_exp = get_area_statistics_from_closed_path(
                self.channel_array, self.path)
        else:
            self.mean_exp, self.max_exp, self.min_exp = 0, 0, 0
