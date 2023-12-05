import math
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path

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

class ZoomRegion(ChannelRegion):
    """
    This class defines a channel region created using the zoom feature
    """
    def __init__(self, channel_array, coordinates):
        super().__init__(channel_array, coordinates)
        self.required_keys = ['xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]']
        if all([elem in self.coordinate_dict] for elem in self.required_keys):
            try:
                assert all([elem >= 0 for elem in self.coordinate_dict.keys() if isinstance(elem, float)])
                x_range_low = math.ceil(int(self.coordinate_dict['xaxis.range[0]']))
                x_range_high = math.ceil(int(self.coordinate_dict['xaxis.range[1]']))
                y_range_low = math.ceil(int(self.coordinate_dict['yaxis.range[1]']))
                y_range_high = math.ceil(int(self.coordinate_dict['yaxis.range[0]']))
                assert x_range_high >= x_range_low
                assert y_range_high >= y_range_low

                self.mean_exp, self.max_exp, self.min_exp = get_area_statistics_from_rect(self.channel_array,
                                                                                          x_range_low,
                                                                                          x_range_high,
                                                                                          y_range_low, y_range_high)
            except AssertionError:
                self.mean_exp, self.max_exp, self.min_exp = 0, 0, 0


class RectangleRegion(ChannelRegion):
    """
    This class defines a channel region created using by drawing a rectangle
    """
    def __init__(self, channel_array, coordinates, first_draw=True):
        super().__init__(channel_array, coordinates)
        self.required_keys = ('x0', 'x1', 'y0', 'y1') if first_draw else ('shapes[1].x0', 'shapes[1].x1',
                                                                          'shapes[1].y0', 'shapes[1].y1')
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
            except AssertionError:
                self.mean_exp, self.max_exp, self.min_exp = 0, 0, 0

class FreeFormRegion(ChannelRegion):
    """
    This class defines a channel region created by drawing a freeform SVG path shape
    """

    def __init__(self, channel_array, coordinates):
        super().__init__(channel_array, coordinates)
        self.path = self.coordinate_dict['path'] if 'path' in self.coordinate_dict else None
        if self.path is not None:
            self.mean_exp, self.max_exp, self.min_exp = get_area_statistics_from_closed_path(
                self.channel_array, self.path)
        else:
            self.mean_exp, self.max_exp, self.min_exp = 0, 0, 0
