"""Module containing utility functions for applying RGB gradient filters at the pixel level
to individual channel arrays
"""
from functools import partial
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize)

class GradientNotFoundError(Exception):
    """
    Raise when an invalid gradient is passed to `ChannelGradient`
    """

SPECTRUM_COLS = {"blue_gold":
    [(0, "black"),
        (1e-6, "blue"),
        (0.5, "white"),
        (1, "gold")],
    # jet just appears as the inverse of rainbow
    "jet": [(0, "black"),
         (1e-6, "red"),
         (0.25, "yellow"),
        (0.5, "green"),
         (0.75, 'cyan'),
        (1, "blue")]}

def apply_map(tile_array: np.array, mapping: LinearSegmentedColormap):
    # tile_array = np.array(Image.fromarray(tile_array).convert('L')) if (
    #         len(tile_array.shape) > 2) else tile_array

    # Normalize the image to [0, 1]
    norm = Normalize(vmin=tile_array.min(), vmax=tile_array.max())
    normalized_image = norm(tile_array)

    # Apply the colormap
    colored_image = mapping(normalized_image)
    # Ensure that no expression is shown as black
    colored_image[tile_array == 0] = [0, 0, 0, 0]

    # Remove the alpha channel (4th channel) if not needed
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

class ChannelGradient:
    """
    Apply a grey scale or RGB color map gradient oto the expression values for a single
    channel array. Colormaps include rainbow, blue gold, and jet (rainbow reversed).

    :param channel_array: The 2D image array to process

    :return: None
    """

    spectrum = {"rainbow": plt.cm.rainbow,
                "blue_gold": LinearSegmentedColormap.from_list("custom_cmap",
                            SPECTRUM_COLS['blue_gold']),
                "jet": LinearSegmentedColormap.from_list("custom_cmap",
                            SPECTRUM_COLS['jet']),
                "greyscale": None}

    def __init__(self, channel_array: np.array):
        self.array = channel_array

    @staticmethod
    def _greyscale(arr: np.array):
        """
        Apply the greyscale gradient to the channel array passed

        :param arr: Individual channel array to process

        :return: Greyscale channel array
        """
        return arr.astype(np.float32) if len(arr.shape) < 3 else \
            np.array(Image.fromarray(arr).convert('L')).astype(np.float32)

    def _rainbow(self, arr: np.array):
        """
        Apply the rainbow gradient to the channel array passed

        :param arr: Individual channel array to process

        :return: Recoloured channel array with expression gradient as rainbow
        """
        return apply_map(arr, self.spectrum['rainbow'])

    def _blue_gold(self, arr: np.array):
        """
        Apply the blue-gold gradient to the channel array passed

        :param arr: Individual channel array to process

        :return: Recoloured channel array with expression gradient as blue-gold
        """
        return apply_map(arr, self.spectrum['blue_gold'])

    def _jet(self, arr: np.array):
        """
        Apply the jet rainbow gradient to the channel array passed

        :param arr: Individual channel array to process

        :return: Recoloured channel array with expression gradient as jet
        """
        return apply_map(arr, self.spectrum['jet'])


    def apply_gradient(self, gradient: str="greyscale"):
        """
        Apply a gradient to a channel array

        :param gradient: String identifier for the gradient to apply

        :return: Channel array with the gradient applied as either greyscale float32 or RGB
        """
        try:
            return partial(getattr(self,
            f'_{gradient.replace(" ", "_")}'))(self.array)
        except AttributeError:
            raise GradientNotFoundError(f"{gradient} is not a valid gradient."
                            f" Please use one of {list(self.spectrum.keys())}")
