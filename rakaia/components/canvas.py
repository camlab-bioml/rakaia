from typing import Union
import math
import numpy as np
import plotly.graph_objs as go
import cv2
import plotly.express as px
from PIL import Image
from plotly.graph_objs.layout import YAxis, XAxis
import pandas as pd
from skimage import measure
from rakaia.parsers.object import validate_coordinate_set_for_image
from rakaia.utils.cluster import get_cluster_proj_id_column
from rakaia.utils.object import generate_greyscale_grid_array
from rakaia.inputs.pixel import (
    add_scale_value_to_figure,
    set_x_axis_placement_of_scalebar,
    default_canvas_margins)
from rakaia.utils.pixel import (
    per_channel_intensity_hovertext,
    get_additive_image,
    apply_filter_to_array,
    create_new_coord_bounds)
from rakaia.utils.object import generate_mask_with_cluster_annotations
from rakaia.utils.shapes import is_cluster_annotation_circle, is_bad_shape
from rakaia.utils.roi import subset_mask_outline_using_cell_id_list

def set_pixel_ratio(pixel_ratio: Union[int, float]=1) -> Union[int, float]:
    """
    Check whether the provided pixel ratio is appropriate (not None and greater than 0)
    Otherwise, set the default pixel ratio to 1
    """
    return pixel_ratio if pixel_ratio is not None and pixel_ratio > 0 else 1

class CanvasImage:
    """
    Generates a canvas `go.Figure` with the current selected channels and various UI configurations for masking,
    annotations, cluster projection, etc.

    :param canvas_layers: dictionary of current ROI channel images. Each key is a biomarker unique ID, and the value
        is the cached RGB image for that layer. If the layer doesn't exist, the marker has not yet been requested
        for the blend.
    :param data_selection: string representation of the current ROI selection
    :param currently_selected: List of channels selected for canvas blend
    :param mask_config: dictionary of imported session mask arrays
    :param mask_selection: String identifier for the mask applied to the canvas
    :param mask_blending_level: Opacity setting for the mask relative to the underlying channel blend. Assumes an integer
        or float value between 0 and 100
    :param overlay_grid: Whether or not to overlay a line grid
    :param mask_toggle: Whether ot overlay a mask
    :param  add_mask_boundary: When a mask is applied, provide a white boundary outline of the objects, regardless of the
        mask opacity.
    :param invert_annot: Whether to invert the scalebar and legend along the x-axis
    :param cur_graph: If a current `go.Figure` graph exists, supply it to retain the `uirevision` variable
    :param pixel_ratio: Integer or float specifying the number of pixels per micron for the image. Default for
        multiplexed imaging is 1.
    :param legend_text: Text supplying the colour-labelled channel list to project over the image
    :param toggle_scalebar: Whether to include the scalebar or not
    :param legend_size: Integer or float size specification for the legend and scalebar relative to the canvas
    :param toggle_legend: Whether to show the legend or not
    :param add_cell_id_hover: If the mask is applied, include the object ID when hovering on the canvas
    :param show_each_channel_intensity: Show the individual raw intensity values from the greyscale array for each
        channel in the hover template
    :param raw_data_dict: Dictionary of the raw greyscale channel images for the current ROI
    :param aliases: Dictionary matching internal channel keys to labels, derived from edited metadata
    :param global_apply_filter: Whether to apply a global gaussian or median filter to the blended channel image
    :param global_filter_type: Specify either a gaussian or median global blur
    :param global_filter_val: Specify a global filter kernel size for median or gaussian blur
    :param global_filter_sigma: If using gaussian blur, set the sigma value for the global filter
    :param apply_cluster_on_mask: Whether to include cluster assignments in the mask or not
    :param cluster-assignments_dict: Dictionary matching cluster categories to RGB assignments inside the mask
    :param cluster_frame: `pd.DataFrame` linking object IDs to cluster assignments
    :param cluster_type: Specify how the clusters should be rendered over the image. Options are `mask` or `circles`.
        Default is `mask`
    :param custom_scale_val: Set a custom scalebar length for the canvas at the un-zoomed level By default, the canvas
        will auto-generate a scalebar length that is 10% of the x-axis in pixels.
    :param apply_gating: Whether or not to apply gating to the mask
    :param gating_cell_id_list: Use a list of mask objects to gate the mask
    :param annotation_color: Specify the color of the scalebar. Options are "white" (default) or "black"
    :param cluster_assignment_selection: Pass a subset of cluster categories to show in the mask
    :return: None
    """
    def __init__(self, canvas_layers: dict, data_selection: str, currently_selected: list,
                 mask_config: dict, mask_selection: str, mask_blending_level: int,
                 overlay_grid: Union[list, bool], mask_toggle: Union[bool, str], add_mask_boundary: Union[bool, str],
                 invert_annot: Union[bool, str], cur_graph: Union[go.Figure, dict], pixel_ratio: Union[int, float, None],
                 legend_text: str, toggle_scalebar: Union[bool, str, list], legend_size: Union[int, float],
                 toggle_legend: Union[bool, str, list], add_cell_id_hover: Union[bool, str, list],
                 show_each_channel_intensity: Union[bool, str, list], raw_data_dict: dict,
                 aliases: dict, global_apply_filter: Union[bool, str, list]=False, global_filter_type: str="median",
                 global_filter_val: Union[int, float]=3, global_filter_sigma: Union[int, float]=1.0,
                 apply_cluster_on_mask: Union[bool, str, list]=False, cluster_assignments_dict: dict=None,
                 cluster_cat: str=None, cluster_frame: Union[pd.DataFrame, dict]=None, cluster_type: str="mask",
                 custom_scale_val: int=None, apply_gating: bool=False, gating_cell_id_list: list=None,
                 annotation_color: str="white", cluster_assignment_selection: Union[list, None]=None):
        self.canvas_layers = canvas_layers
        self.data_selection = data_selection
        self.currently_selected = currently_selected
        self.mask_config = mask_config
        self.mask_selection = mask_selection
        self.mask_blending_level = mask_blending_level
        self.overlay_grid = overlay_grid
        self.mask_toggle = mask_toggle
        self.add_mask_boundary = add_mask_boundary
        self.invert_annot = invert_annot
        try:
            self.cur_graph = CanvasLayout(cur_graph).get_fig()
        except (KeyError, TypeError):
            self.cur_graph = cur_graph
        self.pixel_ratio = set_pixel_ratio(pixel_ratio)
        self.legend_text = legend_text
        self.toggle_scalebar = toggle_scalebar
        self.legend_size = legend_size
        self.toggle_legend = toggle_legend
        self.add_cell_id_hover = add_cell_id_hover
        self.show_each_channel_intensity = show_each_channel_intensity
        self.raw_data_dict = raw_data_dict
        self.aliases = aliases
        self.global_apply_filter = global_apply_filter
        self.global_filter_type = global_filter_type
        self.global_filter_val = global_filter_val
        self.global_filter_sigma = global_filter_sigma if global_filter_sigma is not None else 1
        self.apply_cluster_on_mask = apply_cluster_on_mask
        self.cluster_assignments_dict = cluster_assignments_dict
        self.cluster_cat = cluster_cat
        self.cluster_frame = cluster_frame
        self.cluster_type = cluster_type
        self.custom_scale_val = custom_scale_val
        self.apply_gating = apply_gating
        self.gating_cell_id_list = gating_cell_id_list
        self.annotation_color = annotation_color
        self.uirevision_status = True
        self.get_previous_uirevision()
        self.cluster_selection = cluster_assignment_selection

        image = get_additive_image(self.canvas_layers[self.data_selection], self.currently_selected) if \
            len(self.currently_selected) > 1 else \
            self.canvas_layers[self.data_selection][self.currently_selected[0]].astype(np.float32)
        image = apply_filter_to_array(image, self.global_apply_filter, self.global_filter_type, self.global_filter_val,
                                      self.global_filter_sigma)
        image = np.clip(image, 0, 255)
        self.proportion = 0.1 if self.custom_scale_val is None else \
            float(custom_scale_val / (image.shape[1] * self.pixel_ratio))
        if self.mask_toggle and None not in (self.mask_config, self.mask_selection) and len(self.mask_config) > 0:
            if image.shape[0] == self.mask_config[self.mask_selection]["array"].shape[0] and \
                    image.shape[1] == self.mask_config[self.mask_selection]["array"].shape[1]:
                mask_level = float(self.mask_blending_level / 100) if self.mask_blending_level is not None else 1
                if self.apply_cluster_on_mask and None not in (self.cluster_assignments_dict,
                        self.cluster_frame, self.cluster_cat) and \
                        self.data_selection in self.cluster_assignments_dict.keys() and \
                        self.cluster_cat in self.cluster_assignments_dict[self.data_selection] and self.cluster_type == 'mask':
                    annot_mask = generate_mask_with_cluster_annotations(self.mask_config[self.mask_selection]["raw"],
                                self.cluster_frame[self.data_selection],
                                self.cluster_assignments_dict[self.data_selection][self.cluster_cat],
                                use_gating_subset=self.apply_gating,
                                gating_subset_list=self.gating_cell_id_list,
                                obj_id_col= get_cluster_proj_id_column(self.cluster_frame[self.data_selection]),
                                cluster_option_subset=cluster_assignment_selection,
                                cluster_col=self.cluster_cat)
                    annot_mask = annot_mask if annot_mask is not None else \
                        np.where(self.mask_config[self.mask_selection]["array"].astype(np.uint8) > 0, 255, 0)
                    image = cv2.addWeighted(image.astype(np.uint8), 1, annot_mask.astype(np.uint8), mask_level, 0)
                else:
                    # set the mask blending level based on the slider, by default use an equal blend
                    mask = self.mask_config[self.mask_selection]["array"].astype(np.uint8)
                    mask = self.apply_gating_to_canvas_mask_image(mask)
                    mask = np.where(mask > 0, 255, 0)
                    image = cv2.addWeighted(image.astype(np.uint8), 1, mask.astype(np.uint8), mask_level, 0)
                image = self.overlay_mask_outline_on_mask_image(image)


        image = self.overlay_grid_on_additive_image(image)
        self.image = image
        self.canvas = px.imshow(Image.fromarray(image.astype(np.uint8)), binary_string=True,
                                # currently set to lowest possible compression level for speed
                                binary_compression_level=1)

    def get_previous_uirevision(self):
        """
        # try to get the uirevision status from the current graph if it exists.
        Two possible truthy values are toggled when shapes are cleared
        :return: None
        """
        if self.cur_graph and 'layout' in self.cur_graph and 'uirevision' in self.cur_graph['layout']:
            self.uirevision_status = self.cur_graph['layout']['uirevision']

    def overlay_grid_on_additive_image(self, image: Union[np.array, np.ndarray]) -> np.array:
        """
        Apply evenly spaced gridlines every 100 pixels over the current canvas

        :param image: Numpy array image of the current canvas blend

        :return: Numpy array with white pixels as lines every 100 pixels
        """
        if self.overlay_grid:
            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    generate_greyscale_grid_array((image.shape[0],
                                    image.shape[1])).astype(np.uint8), 1, 0)
        return image

    def apply_gating_to_canvas_mask_image(self, mask: Union[np.array, np.ndarray]) -> np.array:
        """
        Apply gating to the current mask

        :param mask: Numpy array for the current mask selection
        :return: Numpy array of the mask, subset with a mask outline using the class initialize object gating list
        """
        if self.apply_gating:
            mask = subset_mask_outline_using_cell_id_list(self.mask_config[self.mask_selection]["raw"],
                                                          self.mask_config[self.mask_selection]["raw"],
                                                          self.gating_cell_id_list).astype(np.uint8)
        return mask

    def overlay_mask_outline_on_mask_image(self, image: Union[np.array, np.ndarray]) -> np.array:
        """
        Add the subset mask outline using subset objects to the canvas image

        :param image: Numpy array image of the current canvas blend
        :return: Numpy array of the current canvas with the subset mask project over using the class initialized
            mask attributes (blending level, etc.)
        """
        if self.add_mask_boundary and self.mask_config[self.mask_selection]["boundary"] is not None:
            # add the border of the mask after converting back to greyscale to derive the conversion
            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    self.mask_config[self.mask_selection]["boundary"].astype(np.uint8), 1, 0)
        return image

    def generate_canvas(self) -> Union[go.Figure, dict]:
        """
        Convert the blended channel image into a `go.Figure` object using `px.imshow`
        Applies the `px.imshow` function to the RGB array, and adds scalebar and legend annotations

        :return: `go.Figure` object or dictinary representation of the object
        """
        x_axis_placement = set_x_axis_placement_of_scalebar(self.image.shape[1], self.invert_annot)
        # if the current graph already has an image, take the existing layout and apply it to the new figure
        # otherwise, set the uirevision for the first time
        # fig = add_scale_value_to_figure(fig, image_shape, x_axis_placement)
        # do not update if there is already a hover template as it will be too slow
        # scalebar is y = 0.06
        # legend is y = 0.05
        hover_template_exists = 'data' in self.cur_graph and 'customdata' in self.cur_graph['data'] and \
                                self.cur_graph['data']['customdata'] is not None
        if self.current_canvas_exists(hover_template_exists):
            # self.uirevision_status = self.cur_graph['layout']['uirevision']
            try:
                fig = self.transfer_canvas_data_to_existing_canvas()
                # del cur_graph
            # key error could happen if the canvas is reset with no layers, so rebuild from scratch
            except (KeyError, TypeError, ValueError):
                fig = self.canvas
                fig['layout']['uirevision'] = self.uirevision_status

                if self.toggle_scalebar:
                    fig = add_scale_value_to_figure(fig, self.get_shape(), font_size=self.legend_size,
                                                    x_axis_left=x_axis_placement, pixel_ratio=self.pixel_ratio,
                                                    invert=self.invert_annot, proportion=self.proportion,
                                                    scale_color=self.annotation_color)

                fig = go.Figure(fig)
                fig = self.set_default_canvas_layout(fig)
                fig.update_layout(hovermode="x")
        else:
            fig = self.canvas
            # del cur_graph
            # if making the fig for the first time, set the uirevision
            fig['layout']['uirevision'] = self.uirevision_status

            if self.toggle_scalebar:
                fig = add_scale_value_to_figure(fig, self.get_shape(), font_size=self.legend_size,
                                                x_axis_left=x_axis_placement, pixel_ratio=self.pixel_ratio,
                                                invert=self.invert_annot, proportion=self.proportion,
                                                scale_color=self.annotation_color)

            fig = go.Figure(fig)
            fig = self.set_default_canvas_layout(fig)
            fig.update_layout(hovermode="x")

        fig = go.Figure(fig)
        fig.update_layout(newshape=dict(line=dict(color="white")))

        # set how far in from the lefthand corner the scale bar and colour legends should be
        # higher values mean closer to the centre
        fig = self.add_canvas_legend_text(fig, x_axis_placement)

        # set the x-axis scale placement based on the size of the image for adding a scale bar
        fig = self.add_canvas_scalebar(fig, x_axis_placement)

        fig = self.add_canvas_hover_template(fig)
        # to remove the hover template completely
        # fig.update_traces(hovertemplate=None, hoverinfo='skip')
        return fig.to_dict()

    def get_shape(self) -> tuple:
        """
        Get the tuple representing the current canvas image shape

        :return: tuple of the current dimensions (Image height, Image width)
        """
        return self.image.shape

    def get_image(self) -> np.array:
        """
        Get the RGB blended image array for the current canvas

        :return: Numpy RGB blended image array for the current canvas
        """
        return self.image

    def current_canvas_exists(self, hover_template_exists: bool=False) -> bool:
        """
        Check if the current canvas exists and has revision variables that should be retained

        :return: bool indicating if the current canvas exists with a `uirevision` status that should be retained
        """
        return 'layout' in self.cur_graph and 'uirevision' in self.cur_graph['layout'] and \
                self.cur_graph['layout']['uirevision'] and not hover_template_exists

    def transfer_canvas_data_to_existing_canvas(self) -> Union[go.Figure, dict]:
        """
        Transfer the newly created canvas image to the imported canvas if it has existing parameters such as
        uirevision

        :return: `go.Figure` object or dictionary representing the object with current annotations and `uirevision` status
            applied from the existing graph
        """
        # fig['layout'] = cur_graph['layout']
        self.cur_graph['data'] = self.canvas['data']
        # if taking the old layout, remove the current legend and remake with the new layers
        # imp: do not remove the current scale bar value if its there
        if 'annotations' in self.cur_graph['layout'] and len(self.cur_graph['layout']['annotations']) > 0:
            self.cur_graph['layout']['annotations'] = [annotation for annotation in \
                                                       self.cur_graph['layout']['annotations'] if \
                                                       annotation['y'] == 0.06 and self.toggle_scalebar]
        if 'shapes' in self.cur_graph['layout'] and len(self.cur_graph['layout']['shapes']):
            self.cur_graph['layout']['shapes'] = [shape for shape in self.cur_graph['layout']['shapes'] if \
                                                  shape['type'] != 'line']
        return self.cur_graph

    def add_canvas_legend_text(self, fig: go.Figure, x_axis_placement: Union[int, float]) -> Union[go.Figure, dict]:
        """
        Add canvas legend text using a specified text size and x axis placement
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary

        :param fig: Current `go.Figure` or dictionary of the object for the current canvas
        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the legend text

        :return: go.Figure` object or dictionary representing the object with the canvas legend text added
        """
        if self.legend_text != '' and self.toggle_legend:
            fig.add_annotation(text=self.legend_text, font={"size": self.legend_size + 1}, xref='paper',
                               yref='paper',
                               x=(1 - x_axis_placement),
                               # xanchor='right',
                               y=0.05,
                               # yanchor='bottom',
                               bgcolor="black",
                               showarrow=False)
        return fig

    def add_canvas_hover_template(self, fig: go.Figure) -> Union[go.Figure, dict]:
        """
        Add a hover template to the current canvas object

        :param fig: Current `go.Figure` or dictionary of the object for the current canvas

        :return: go.Figure` object or dictionary representing the object with the hover template applied as a data slot
        """
        # the masking mask ID get priority over the channel intensity hover
        if self.mask_toggle and None not in (self.mask_config, self.mask_selection) and len(self.mask_config) > 0 and \
                self.add_cell_id_hover:
            try:
                # fig.update(data=[{'customdata': None}])
                fig.update(data=[{'customdata': self.mask_config[self.mask_selection]["hover"]}])
                new_hover = per_channel_intensity_hovertext(["mask ID"])
            except KeyError:
                new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"

        elif self.show_each_channel_intensity:
            # fig.update(data=[{'customdata': None}])
            hover_stack = np.stack(tuple(self.raw_data_dict[self.data_selection][elem] for \
                                         elem in self.currently_selected),
                                   axis=-1)
            fig.update(data=[{'customdata': hover_stack}])
            # set the labels for the hover from the aliases
            hover_labels = []
            for label in self.currently_selected:
                if label in self.aliases.keys():
                    hover_labels.append(self.aliases[label])
                else:
                    hover_labels.append(label)
            new_hover = per_channel_intensity_hovertext(hover_labels)
        else:
            fig.update(data=[{'customdata': None}])
            new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"
        fig.update_traces(hovertemplate=new_hover)
        return fig

    def add_canvas_scalebar(self, fig: go.Figure, x_axis_placement: Union[int, float]) -> Union[go.Figure, dict]:
        """
        Add a canvas scalebar with a set bar width of 2 and a number size set by the user
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary

        :param fig: Current `go.Figure` or dictionary of the object for the current canvas
        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the scalebar

        :return: go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        if self.toggle_scalebar:
            # set the x0 and x1 depending on if the bar is inverted or not
            x_0 = x_axis_placement if not self.invert_annot else (x_axis_placement - self.proportion)
            x_1 = (x_axis_placement + self.proportion) if not self.invert_annot else x_axis_placement
            fig.add_shape(type="line",
                          xref="paper", yref="paper",
                          x0=x_0, y0=0.05, x1=x_1,
                          y1=0.05, line=dict(color=self.annotation_color, width=2))
        return fig

    @staticmethod
    def set_default_canvas_layout(fig: go.Figure) -> Union[go.Figure, dict]:
        """
        Set the default canvas margin and grid layout

        :param fig: Current `go.Figure` or dictionary of the object for the current canvas

        :return: go.Figure` object or dictionary representing the object with the margin and grid line layout applied
        """
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          newshape=dict(line=dict(color="white")),
                          xaxis=XAxis(showticklabels=False, domain=[0, 1]),
                          yaxis=YAxis(showticklabels=False, domain=[0, 1]),
                          margin=default_canvas_margins())
        return fig


class CanvasLayout:
    """
    Represents a set of layout manipulations for the image canvas. It is distinct from
    CanvasImage in that it doesn't manipulate the underlying image, but rather the layout and UI components
    of the attributes projected on top of the data. It expected a `go.Figure` object or a dictionary representing
    a `go.Figure` object as input

    :param figure = `go.Figure` for the current canvas

    :return:
        None
    """
    def __init__(self, figure: Union[dict, go.Figure]):

        self.cur_annotations = []
        self.cur_shapes = []
        figure = self.parse_out_bad_shapes(figure)
        try:
            figure['layout']['yaxis']['domain'] = [0, 1]
            figure['layout']['xaxis']['domain'] = [0, 1]
        except KeyError:
            pass

        self.figure = figure
        self.set_current_annotations()
        self.set_current_shapes()

        for shape in self.cur_shapes:
            if 'label' in shape and 'texttemplate' in shape['label']:
                shape['label'] = {}

    @staticmethod
    def parse_out_bad_shapes(figure) -> Union[go.Figure, dict]:
        """
        Remove any malformed shapes from the canvas

        :param figure: Current canvas as `go.Figure`
        :return: `go.Figure`` with malformed shapes in the layout removes
        """
        if 'layout' in figure and 'shapes' in figure['layout'] and \
                len(figure['layout']['shapes']) > 0 and not \
                isinstance(figure['layout']['shapes'], tuple):
            figure['layout']['shapes'] = [shape for shape in figure['layout']['shapes'] if \
                                          shape and not is_bad_shape(shape)]
        return figure

    def set_current_annotations(self):
        """
        Set and retain the annotations from the current canvas. Includes the legend and scalebar text.

        :return: None
        """
        if 'layout' in self.figure and 'annotations' in self.figure['layout'] and \
                len(self.figure['layout']['annotations']) > 0 and not \
                isinstance(self.figure['layout']['annotations'], tuple):
            self.cur_annotations = [annot for annot in self.figure['layout']['annotations'] if annot is not None]

    def set_current_shapes(self):
        """
        Set and retain the shapes from the current canvas. Drawn shapes will be kept through canvas updates.

        :return: None
        """
        if 'layout' in self.figure and 'shapes' in self.figure['layout'] and \
                len(self.figure['layout']['shapes']) > 0 and not \
                isinstance(self.figure['layout']['shapes'], tuple):
            self.cur_shapes = [shape for shape in self.figure['layout']['shapes'] if shape and \
                               'type' in shape and not is_bad_shape(shape)]

    def get_fig(self) -> Union[go.Figure, dict]:
        """
        Return the modified canvas figure

        :return:
            Current `go.Figure` or dictionary of the object for the current canvas
        """
        self.figure['layout']['shapes'] = self.cur_shapes
        self.figure['layout']['annotations'] = self.cur_annotations
        return self.figure

    def add_scalebar(self, x_axis_placement, invert_annot, pixel_ratio, image_shape, legend_size,
                     proportion=0.1, annotation_color: str="white") -> Union[go.Figure, dict]:
        """
        Add a canvas scalebar with a set bar width of 2 and a number size set by the user
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary

        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the scalebar
        :param invert_annot: Boolean for whether or not to invert the scalebar nad legend text along the x-axis
        :param pixel_ratio: ratio of pixels to micron. FOr most imaging experiments, should be set to 1.
        :param image_shape: Tuple of numpy array shape
        :param legend_size: Integer or float of the legend font size relative to the canvas object
        :param proportion: What proportion of the x-axis width should the length of the scalebar be. By default,
            it will span 10% of the x-axis
        :param annotation_color: Specify the color of the scalebar. Options are "white" (default) or "black"

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        pixel_ratio = set_pixel_ratio(pixel_ratio)
        try:
            proportion = float(proportion / pixel_ratio)
        except ZeroDivisionError:
            pass
        fig = go.Figure(self.figure)
        fig.update_layout(newshape=dict(line=dict(color="white")))
        # default length is 0.1 (10% of the canvas), but want to make adjustable
        # set the x0 and x1 depending on if the bar is inverted or not
        x_0 = x_axis_placement if not invert_annot else (x_axis_placement - proportion)
        x_1 = (x_axis_placement + proportion) if not invert_annot else x_axis_placement
        fig.add_shape(type="line",
                      xref="paper", yref="paper",
                      x0=x_0, y0=0.05, x1=x_1,
                      y1=0.05, line=dict(color=annotation_color, width=2))

        try:
            high = max(self.figure['layout']['xaxis']['range'][1],
                       self.figure['layout']['xaxis']['range'][0])
            low = min(self.figure['layout']['xaxis']['range'][1],
                      self.figure['layout']['xaxis']['range'][0])
            x_range_high = math.ceil(int(high))
            x_range_low = math.floor(int(low))
            if x_range_high < x_range_low: raise AssertionError
            custom_scale_val = int(float(math.ceil(int(proportion *
                                (x_range_high - x_range_low))) + 1) * float(pixel_ratio))
        except (KeyError, TypeError, AssertionError):
            custom_scale_val = None

        fig = add_scale_value_to_figure(fig, image_shape, scale_value=custom_scale_val,
                                font_size=legend_size, x_axis_left=x_axis_placement, invert=invert_annot,
                                proportion=proportion, scale_color=annotation_color)

        return fig.to_dict()

    def add_legend_text(self, legend_text, x_axis_placement, legend_size) -> Union[go.Figure, dict]:
        """
        Add legend text to the current canvas

        :param legend_text: String representation of the current channels in the blend with their color designations
        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the scalebar
        :param legend_size: Integer or float of the legend font size relative to the canvas object

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        fig = go.Figure(self.figure)
        fig.update_layout(newshape=dict(line=dict(color="white")))
        if legend_text != '':
            fig.add_annotation(text=legend_text, font={"size": legend_size + 1}, xref='paper',
                                   yref='paper',
                                   x=(1 - x_axis_placement),
                                   # xanchor='right',
                                   y=0.05,
                                   # yanchor='bottom',
                                   bgcolor="black",
                                   showarrow=False)
        return fig.to_dict()

    def toggle_legend(self, toggle_legend: bool, legend_text, x_axis_placement, legend_size) -> Union[go.Figure, dict]:
        """
        Modify the legend text for the figure, or remove the legend

        :param legend_text: String representation of the current channels in the blend with their color designations
        :param toggle_legend: Whether or not to show the channel legend over the canvas
        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the scalebar
        :param legend_size: Integer or float of the legend font size relative to the canvas object
            it will span 10% of the x-axis

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        cur_annotations = [annot for annot in self.cur_annotations if \
                           annot is not None and 'y' in annot and annot['y'] != 0.05]
        self.figure['layout']['annotations'] = cur_annotations
        if not toggle_legend:
            return self.figure
        else:
            return self.add_legend_text(legend_text, x_axis_placement, legend_size)

    def toggle_scalebar(self, toggle_scalebar, x_axis_placement, invert_annot,
                        pixel_ratio, image_shape, legend_size, proportion=0.1,
                        scalebar_color: str="white") -> Union[go.Figure, dict]:
        """
        Add a canvas scalebar with a set bar width of 2 and a number size set by the user
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary

        :param x_axis_placement: Coordinate between 0 and 1 for the x-axis placement for the scalebar
        :param toggle_scalebar: Whether to show the scalebar oevr the canvas or not
        :param invert_annot: Boolean for whether or not to invert the scalebar nad legend text along the x-axis
        :param pixel_ratio: ratio of pixels to micron. FOr most imaging experiments, should be set to 1.
        :param image_shape: Tuple of numpy array shape
        :param legend_size: Integer or float of the scalebar font size relative to the canvas object
        :param proportion: What proportion of the x-axis width should the length of the scalebar be. By default,
                    it will span 10% of the x-axis
        :param scalebar_color: Color of the scalebar relative to the image. default is `white`.

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        pixel_ratio = set_pixel_ratio(pixel_ratio)
        cur_shapes = [shape for shape in self.cur_shapes if \
                      shape not in [None, "None"] and 'type' in shape and shape['type'] \
                      in ['rect', 'path', 'circle'] and not is_bad_shape(shape)]
        cur_annotations = [annot for annot in self.cur_annotations if \
                           annot is not None and 'y' in annot and annot['y'] != 0.06]
        for shape in cur_shapes:
            if 'label' in shape and 'texttemplate' in shape['label']:
                shape['label'] = {}
        self.figure['layout']['annotations'] = cur_annotations
        self.figure['layout']['shapes'] = cur_shapes
        self.figure['layout']['uirevision'] = True if self.figure['layout']['uirevision'] not in [True] else "clear"
        if not toggle_scalebar:
            return self.figure
        return self.add_scalebar(x_axis_placement, invert_annot,
                    pixel_ratio, image_shape, legend_size, proportion, scalebar_color)

    def change_annotation_size(self, legend_size) -> Union[go.Figure, dict]:
        """
        Change the size of the legend and scalebar

        :param legend_size: Integer or float of the font size relative to the canvas object

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        # annotations_copy = self.figure['layout']['annotations'].copy() if not isinstance()
        for annotation in self.cur_annotations:
            # the scalebar is always slightly smaller
            if annotation['y'] == 0.06:
                annotation['font']['size'] = legend_size
            elif annotation['y'] == 0.05 and 'color' in annotation['text']:
                annotation['font']['size'] = legend_size + 1
        self.figure['layout']['annotations'] = [elem for elem in self.figure['layout']['annotations'] if \
                                              elem is not None and 'texttemplate' not in elem]
        return self.figure

    def add_point_annotations_as_circles(self, imported_annotations,
                                         cur_image, circle_size) -> Union[go.Figure, dict]:
        """
        Add a circle for each point annotation in a CSV file. Each annotation is validated against the
        image dimensions in the current canvas to ensure that the annotation lies within the dimensions

        :param imported_annotations: pd.DatdFrame of imported click-point annotations
        :param cur_image: RGB numpy array of the current canvas blend
        :param circle_size: Integer or float specifying the radius size of the circles to be drawn on the canvas

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        imported_annotations = pd.DataFrame(imported_annotations)
        # fig = go.Figure(self.figure)
        for index, row in imported_annotations.iterrows():
            if validate_coordinate_set_for_image(row['x'], row['y'], cur_image):
                self.cur_shapes.append(
                    {'editable': True, 'line': {'color': 'white'}, 'type': 'circle',
                     'x0': (row['x'] - circle_size), 'x1': (row['x'] + circle_size),
                     'xref': 'x', 'y0': (row['y'] - circle_size), 'y1': (row['y'] + circle_size), 'yref': 'y'})
        self.figure['layout']['shapes'] = self.cur_shapes
        return self.figure

    def update_scalebar_zoom_value(self, current_graph_layout, pixel_ratio, proportion=0.1,
                                   scalebar_col: str="white") -> Union[go.Figure, dict]:
        """
        update the scalebar value when zoom is used
        Loop through the annotations to identify the scalebar value when y = 0.06

        :param current_graph_layout: Dictionary of the current zoom parameters of the canvas
        :param pixel_ratio: ratio of pixels to micron. FOr most imaging experiments, should be set to 1.
        :param proportion: What proportion of the x-axis width should the length of the scalebar be. By default,
                    it will span 10% of the x-axis
        :param scalebar_col: Color of the scalebar relative to the image. default is `white`.

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        pixel_ratio = set_pixel_ratio(pixel_ratio)
        try:
            proportion = float(proportion / pixel_ratio)
        except ZeroDivisionError:
            pass
        # find the text annotation that has um in the text and the correct location
        for annotations in self.figure['layout']['annotations']:
            # if 'μm' in annotations['text'] and annotations['y'] == 0.06:
            if annotations['y'] == 0.06:
                x_range_high = 0
                x_range_low = 0
                # use different variables depending on how the ranges are written in the dict
                # IMP: the variables will be written differently after a tab change
                if all(axis in current_graph_layout for axis in ['xaxis.range[0]', 'xaxis.range[1]']):
                    high = max(current_graph_layout['xaxis.range[1]'],
                               current_graph_layout['xaxis.range[0]'])
                    low = min(current_graph_layout['xaxis.range[1]'],
                              current_graph_layout['xaxis.range[0]'])
                    x_range_high = math.ceil(int(high))
                    x_range_low = math.ceil(int(low))
                elif 'xaxis' in self.figure['layout'] and 'range' in self.figure['layout']['xaxis'] and \
                        self.figure['layout']['xaxis']:
                    high = max(self.figure['layout']['xaxis']['range'][1],
                               self.figure['layout']['xaxis']['range'][0])
                    low = min(self.figure['layout']['xaxis']['range'][1],
                              self.figure['layout']['xaxis']['range'][0])
                    x_range_high = math.ceil(int(high))
                    x_range_low = math.floor(int(low))
                if x_range_high < x_range_low: raise AssertionError
                # Enforce that all values must be above 0 for the scale value to render during panning
                scale_val = int(float(math.ceil(int(proportion * (x_range_high - x_range_low))) + 1) * float(
                    pixel_ratio))
                scale_val = scale_val if scale_val > 0 else 1
                scale_annot = str(scale_val) + "μm"
                scale_text = f'<span style="color: {scalebar_col}">{str(scale_annot)}</span><br>'
                # get the index of the list element corresponding to this text annotation
                index = self.figure['layout']['annotations'].index(annotations)
                self.figure['layout']['annotations'][index]['text'] = scale_text

        # fig = go.Figure(self.figure)
        # fig.update_layout(newshape=dict(line=dict(color="white")))
        # return fig
        return self.figure

    def use_custom_scalebar_value(self, custom_scale_val, pixel_ratio, proportion=0.1) -> Union[go.Figure, dict]:
        """
        Specify a custom scalebar length

        :param custom_scale_val: Set a custom scalebar length for the canvas at the un-zoomed level By default, the canvas
            will auto-generate a scalebar length that is 10% of the x-axis in pixels.
        :param pixel_ratio: ratio of pixels to micron. FOr most imaging experiments, should be set to 1.
        :param proportion: What proportion of the x-axis width should the length of the scalebar be. By default,
                    it will span 10% of the x-axis

        :return:
            go.Figure` object or dictionary representing the object with the canvas scalebar added
        """
        # self.figure = strip_invalid_shapes_from_graph_layout(self.figure)
        pixel_ratio = set_pixel_ratio(pixel_ratio)
        for annotations in self.figure['layout']['annotations']:
            # if 'μm' in annotations['text'] and annotations['y'] == 0.06:
            if annotations['y'] == 0.06:
                if custom_scale_val is None:
                    high = max(self.figure['layout']['xaxis']['range'][1],
                               self.figure['layout']['xaxis']['range'][0])
                    low = min(self.figure['layout']['xaxis']['range'][1],
                              self.figure['layout']['xaxis']['range'][0])
                    x_range_high = math.ceil(int(high))
                    x_range_low = math.floor(int(low))
                    if not x_range_high >= x_range_low: raise AssertionError
                    custom_scale_val = int(float(math.ceil(int(proportion *
                                        (x_range_high - x_range_low))) + 1) * float(pixel_ratio))
                else:
                    custom_scale_val = int(float(custom_scale_val) * float(pixel_ratio))
                custom_scale_val = custom_scale_val + 1 if custom_scale_val == 0 else custom_scale_val
                scale_annot = str(custom_scale_val) + "μm"
                scale_text = f'<span style="color: white">{str(scale_annot)}</span><br>'
                # get the index of the list element corresponding to this text annotation
                index = self.figure['layout']['annotations'].index(annotations)
                self.figure['layout']['annotations'][index]['text'] = scale_text
        fig = go.Figure(self.figure)
        fig.update_layout(newshape=dict(line=dict(color="white")))
        return fig.to_dict()

    def clear_improper_shapes(self) -> Union[go.Figure, dict]:
        """
        Remove any malformed canvas shapes that have an empty label slow in the `texttemplate` dictionary

        :return:
            go.Figure` object or dictionary representing the object with the shapes cleared
        """
        new_shapes = []
        for shape in self.cur_shapes:
            if 'label' in shape and 'texttemplate' not in shape['label']:
                shape['label'] = {}
            try:
                if not is_bad_shape(shape):
                    new_shapes.append(shape)
            except KeyError:
                pass
        self.figure['layout']['shapes'] = new_shapes
        return self.figure

    def add_cluster_annotations_as_circles(self, mask, cluster_frame, cluster_assignments,
                                           data_selection, circle_size=2, use_gating: bool=False,
                                           gating_cell_id_list: list=None,
                                           cluster_selection_subset: list=None,
                                           cluster_id_col: str="cluster") -> Union[go.Figure, dict]:
        """
        Add an annotation circle to every mask object in a mask, or in a list of gated objects

        :param mask: a mask with raw object values starting at 1 in numpy int32 form
        :param cluster_frame: `pd.DataFrame` with the columns `cell_id` and `cluster`
        :param cluster_assignments: dictionary of cluster labels corresponding to a hex colour
        :param data_selection: string representation of the current ROI
        :param circle_size: Circle radius for the cluster projection over the mask object
        :param use_gating: Whether to apply gating to the mask
        :param gating_cell_id_list: Use a list of mask objects to gate the mask
        :param cluster_selection_subset: Pass a subset of cluster categories to show in the mask
        :param cluster_id_col: The identifying column for the currently selected cluster category.
        :return:
            go.Figure` object or dictionary representing the object with the cluster annotations added as circles
        """
        object_identify_col = get_cluster_proj_id_column(pd.DataFrame(cluster_frame))
        shapes = []
        if self.cur_shapes:
            # make sure to clear the existing circles first
            shapes = [shape for shape in self.cur_shapes if self.cur_shapes and not ('editable' in shape and
                    not shape['editable'] and 'type' in shape and shape['type'] == 'circle')]
        cluster_frame = pd.DataFrame(cluster_frame)
        cluster_frame = cluster_frame.astype(str)
        ids_use = gating_cell_id_list if (gating_cell_id_list is not None and use_gating) else np.unique(mask)
        clusters_to_use = cluster_selection_subset if cluster_selection_subset is not None else \
            cluster_frame[cluster_id_col].unique().tolist()
        clusters_to_use = [str(clust) for clust in clusters_to_use]
        for mask_id in ids_use:
            try:
                annotation = pd.Series(cluster_frame[cluster_frame[object_identify_col] ==
                                                     str(mask_id)][cluster_id_col]).to_list()
                if annotation and str(annotation[0]) in clusters_to_use:
                    annotation = str(annotation[0])
                    # IMP: each region needs to be subset before region props are computed, or the centroids are wrong
                    subset = np.where(mask == int(mask_id), int(mask_id), 0)
                    region_props = measure.regionprops(subset)
                    for region in region_props:
                        center = region.centroid
                        # boundary[int(center[0]), int(center[1])] = mask_id
                        shapes.append(
                            {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
                             'x0': (int(center[1]) - circle_size), 'x1': (int(center[1]) + circle_size),
                             'xref': 'x', 'y0': (int(center[0]) - circle_size), 'y1': (int(center[0]) + circle_size),
                             'yref': 'y',
                             'fillcolor': cluster_assignments[data_selection][cluster_id_col][annotation]})
            except ValueError:
                pass
        self.figure['layout']['shapes'] = shapes
        return self.figure

    def remove_cluster_annotation_shapes(self) -> Union[go.Figure, dict]:
        """
        Remove the cluster annotation shapes from the canvas.
        These are uniquely recognized as circles that are not editable

        :return:
            go.Figure` object or dictionary representing the object with the cluster circle annotations removed
        """
        new_shapes = []
        for shape in self.cur_shapes:
            if 'editable' not in shape or not is_cluster_annotation_circle(shape):
                new_shapes.append(shape)
        self.figure['layout']['shapes'] = new_shapes
        return self.figure

    def update_coordinate_window(self, current_window, x_request, y_request) -> tuple:
        """
        Specify a new central canvas coordinate while retaining the curent canvas zoom level

        :param current_window: Dictionary of the x and y bounds of the current window to derive the zoom level
        :param x_request: Requested x-coordinate for the center point
        :param y_request: Requested y-coordinate for the center point

        :return:
            go.Figure` object or dictionary representing the object with new zoom window applied as the view range
        """
        # calculate midway distance for each coord. this distance is
        # added on either side of the x and y requests
        new_x_low, new_x_high, new_y_low, new_y_high = create_new_coord_bounds(current_window,
                                                                               x_request,
                                                                               y_request)
        new_layout = {'xaxis.range[0]': new_x_low, 'xaxis.range[1]': new_x_high,
                      'yaxis.range[0]': new_y_high, 'yaxis.range[1]': new_y_low}
        # IMP: for yaxis, need to set the min and max in the reverse order
        fig = go.Figure(data=self.figure['data'], layout=self.figure['layout'])
        shapes = self.figure['layout']['shapes']
        annotations = self.figure['layout']['annotations']
        fig['layout']['shapes'] = None
        fig['layout']['annotations'] = None
        fig.update_layout(xaxis=XAxis(showticklabels=False, range=[new_x_low, new_x_high]),
                          yaxis=YAxis(showticklabels=False, range=[new_y_high, new_y_low]))
        fig.update_layout(newshape=dict(line=dict(color="white")))
        # cur_graph['layout']['xaxis']['domain'] = [0, 1]
        # cur_graph['layout']['dragmode'] = "zoom"
        fig['layout']['shapes'] = shapes
        fig['layout']['annotations'] = annotations
        return fig.to_dict(), new_layout

    def add_click_point_circle(self, x_coord: int=None, y_coord: int=None,
                               circle_size: Union[float, int]=None) -> Union[go.Figure, dict]:
        """
        Add a click point coordinate as a circle shape object

        :param x_coord: x coordinate for the click
        :param y_coord: y coordinate for the click
        :param circle_size: radius of the circle shape to be drawn around the coordinate

        :return:
            go.Figure` object or dictionary representing the object with the coordinate supplied as a circle shape
        """
        self.cur_shapes.append({'editable': True, 'line': {'color': 'white'}, 'type': 'circle',
                                'x0': (x_coord - int(circle_size)), 'x1': (x_coord + int(circle_size)),
                                'xref': 'x', 'y0': (y_coord - int(circle_size)),
                                'y1': (y_coord + int(circle_size)), 'yref': 'y'})
        self.figure['layout']['shapes'] = self.cur_shapes
        return self.figure


def reset_graph_with_malformed_template(graph: Union[go.Figure, dict]) -> Union[go.Figure, dict]:
    """
    Parse a current graph that may have malformed shapes (i.e. a shape with a blank texttemplate in the 'label'
    slot), and return a cleaned graph dictionary object with the drag mode set to zoom

    :param graph: Current canvas in `go.Figure` object format
    :return:
            go.Figure` object or dictionary representing the object with the drag mode reset to zoom
    """
    graph = graph.to_dict() if not isinstance(graph, dict) else graph
    fig = go.Figure(CanvasLayout(graph).get_fig())
    fig.update_layout(dragmode="zoom")
    return fig
