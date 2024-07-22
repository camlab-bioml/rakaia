import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from rakaia.utils.pixel import (
    apply_filter_to_array,
    get_bounding_box_for_svgpath)
from rakaia.utils.object import (
    get_min_max_values_from_zoom_box,
    get_min_max_values_from_rect_box)

class AnnotationPDFWriter:
    """
    An instance of a matplotlib backend pdf writer for a dictionary of image annotations

    :param annotations_dict: Dictionary of annotations for the current ROI
    :param data_selection: String representation of the current session ROI
    :param mask_config: Dictionary of the imported mask options
    identifier, and ROI name
    param dest_dir: Output directory for the PDF
    :param output_file: Filename output for the PDF
    :param blend_dict: Dictionary of the blend parameters for the channel panel
    :param global_apply_filter: Whether or not a global filter has been applied
    :param global_filter_type: String specifying a global gaussian or median blur
    :param global_filter_val: Kernel size for the global gaussian or median blur
    :param global_filter_sigma: If global gaussian blur is applied, set the sigma value
    :return: None
    """
    def __init__(self, dest_dir:str=None, annotations_dict: dict=None, canvas_layers: dict=None,
                 data_selection: str=None, mask_config: dict=None,
                 aliases: dict=None, output_file="annotations.pdf", blend_dict=None,
                 global_apply_filter=False, global_filter_type="median", global_filter_val=3, global_filter_sigma=1):
        self.annotations_dict = annotations_dict
        self.canvas_layers = canvas_layers
        self.data_selection = data_selection
        self.mask_config = mask_config
        self.aliases = aliases
        self.filepath = os.path.join(dest_dir, output_file)
        self.blend_dict = blend_dict
        if self.data_selection in self.annotations_dict and len(self.annotations_dict) > 0:
            self.annotations_dict = {key: value for key, value in annotations_dict[data_selection].items() if \
                                value['type'] not in ['point']}
        self.global_apply_filter = global_apply_filter
        self.global_filter_type = global_filter_type
        self.global_filter_val = global_filter_val
        self.global_filter_sigma = global_filter_sigma

    def generate_annotation_pdf(self):
        """
        Write the annotation PDF to file

        :return: Output PDF filepath.
        """
        if len(self.annotations_dict) > 0:
            with PdfPages(self.filepath) as pdf:
                for key, value in self.annotations_dict.items():
                    if value['type'] in ['zoom', 'rect', 'path']:
                        # the key is the tuple of the coordinates or the svgpath
                        if value['type'] == "zoom":
                            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
                        elif value['type'] == "rect":
                            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))
                        elif value['type'] == "path":
                            x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(key)
                        try:
                            image = sum([np.asarray(self.canvas_layers[self.data_selection][elem]).astype(np.float32) for \
                             elem in value['channels'] if \
                             elem in self.canvas_layers[self.data_selection].keys()]).astype(np.float32)
                            image = apply_filter_to_array(image, self.global_apply_filter, self.global_filter_type,
                                                          self.global_filter_val,
                                                          self.global_filter_sigma)
                            image = np.clip(image, 0, 255)
                        except KeyError:
                            image = None
                        image = self.generate_additive_image_for_pdf(image, value)
                        region = np.array(image[np.ix_(range(int(y_min), int(y_max), 1),
                                               range(int(x_min), int(x_max), 1))]).astype(np.uint8)
                        aspect_ratio = image.shape[1] / image.shape[0]
                        # set height based on the pixel number
                        height = 0.02 * image.shape[1] if 0.02 * image.shape[1] < 30 else 30
                        width = height * aspect_ratio
                        # first value is the width, second is the height
                        fig = plt.figure(figsize=(width, height))
                        fig.tight_layout()
                        # ax = fig.add_subplot(111)
                        # plt.axes((.1, .4, .8, .5))
                        ax = fig.add_axes((0, .4, 1, 0.5))
                        ax.imshow(region, interpolation='nearest')
                        ax.set_title(value['title'], fontsize=(width + 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        x_dims = float(x_max) - float(x_min)
                        y_dims = float(y_max) - float(y_min)
                        patches = self.pdf_patches_legend(value)
                        body = str(value['body']).replace(r'\n', '\n')
                        description = "Description:\n" + body + "\n\n" + "" \
                                "Region dimensions: " + str(int(x_dims)) + "x" + str(int(y_dims))
                        text_offset = .3 if height < 25 else .2
                        fig.text(.15, text_offset, description, fontsize=width)
                        fig.legend(handles=patches, fontsize=width, title='Channel List', title_fontsize=(width + 5))
                        # ax.set_xlabel(description, fontsize=25)
                        # y_offset = 0.95
                        # plt.figtext(0.01, 1, "Channels", size=16)
                        # for channel in channel_list:
                        #     plt.figtext(0.01, y_offset, channel, size=14)
                        #     y_offset -= 0.05
                        pdf.savefig()
            return self.filepath
        return None

    def generate_additive_image_for_pdf(self, image, value):
        """
        Produce an additive blend image for a specific region

        :param image: Blended RGB array for a region with the currently selected channels
        :param value: Annotation dictionary specifying the annotation mask parameters
        :return: Additive blended numpy array with a mask overlaid (if enabled)
        """
        if value['use_mask'] and None not in (self.mask_config, value['mask_selection']) and \
                len(self.mask_config) > 0:
            if image.shape[0] == self.mask_config[value['mask_selection']]["array"].shape[0] and \
                    image.shape[1] == self.mask_config[value['mask_selection']]["array"].shape[1]:
                # set the mask blending level based on the slider, by default use an equal blend
                mask_level = float(value['mask_blending_level'] / 100) if \
                    value['mask_blending_level'] is not None else 1
                image = cv2.addWeighted(image.astype(np.uint8), 1,
                                        self.mask_config[value['mask_selection']]["array"].astype(np.uint8),
                                        mask_level, 0)
            if value['add_mask_boundary'] and \
                    self.mask_config[value['mask_selection']]["boundary"] is not None:
                # add the border of the mask after converting back to greyscale to derive the conversion
                reconverted = np.array(Image.fromarray(self.mask_config[value['mask_selection']][
                                                           "boundary"]).convert('RGB'))
                image = cv2.addWeighted(image.astype(np.uint8), 1, reconverted.astype(np.uint8), 1, 0)
        return image

    def pdf_patches_legend(self, value):
        """
        Generate a channel legend text as a matplotlib patch

        :param value: Annotation dictionary specifying the annotation channel blend parameters
        :return: list of channels and their colors in patch format.
        """
        patches = []
        for channel in value['channels']:
            label = self.aliases[channel] if channel in self.aliases.keys() else channel
            if self.blend_dict is not None:
                try:
                    col_use = self.blend_dict[channel]['color']
                except KeyError:
                    col_use = 'white'
            else:
                col_use = 'white'
            patches.append(mpatches.Patch(color=col_use, label=label))
        return patches
