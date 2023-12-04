from abc import ABC, abstractmethod


class ChannelRegion(ABC):
    """
    This abstract class defines a region for a particular channel
    """
    @abstractmethod
    def __init__(self, channel_array, coordinate_dict):
        self.channel_array = channel_array
        self.coordinate_dict = coordinate_dict

    @abstractmethod
    def compute_pixel_mean(self):
        """
        Compute the mean pixel intensity of the channel region
        """
        pass

    @abstractmethod
    def compute_pixel_min(self):
        """
        Compute the min pixel intensity of the channel region
        """
        pass

    @abstractmethod
    def compute_pixel_max(self):
        """
        Compute the max pixel intensity of the channel region
        """
        pass

class ZoomRegion(ChannelRegion, ABC):
    """
    This class defines a channel region created using the zoom feature
    """
    def __init__(self):
        super().__init__()
