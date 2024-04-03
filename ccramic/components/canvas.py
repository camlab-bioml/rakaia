from typing import Union
import numpy as np
import plotly.graph_objs as go
import cv2
import plotly.express as px
from PIL import Image
from ccramic.parsers.cell_level_parsers import validate_coordinate_set_for_image
from ccramic.utils.cell_level_utils import generate_greyscale_grid_array
from ccramic.inputs.pixel_level_inputs import (
    add_scale_value_to_figure,
    set_x_axis_placement_of_scalebar)
from ccramic.utils.pixel_level_utils import (
    per_channel_intensity_hovertext,
    get_additive_image,
    apply_filter_to_array,
    create_new_coord_bounds)
from ccramic.utils.cell_level_utils import generate_mask_with_cluster_annotations
from plotly.graph_objs.layout import YAxis, XAxis
from ccramic.utils.shapes import is_cluster_annotation_circle, is_bad_shape
from ccramic.utils.roi_utils import subset_mask_outline_using_cell_id_list
import pandas as pd
import math
from skimage import measure

class CanvasImage:
    """
    This class generates a canvas `go.Figure` with the current selected channels and various
    UI configurations
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
                 cluster_frame: Union[pd.DataFrame, dict]=None, cluster_type: str="mask", custom_scale_val: int=None,
                 apply_gating: bool=False, gating_cell_id_list: list=None, annotation_color: str="white"):
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
        self.pixel_ratio = pixel_ratio if pixel_ratio is not None and pixel_ratio > 0 else 1
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
        self.cluster_frame = cluster_frame
        self.cluster_type = cluster_type
        self.custom_scale_val = custom_scale_val
        self.apply_gating = apply_gating
        self.gating_cell_id_list = gating_cell_id_list
        self.annotation_color = annotation_color

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
                if self.apply_cluster_on_mask and None not in (self.cluster_assignments_dict, self.cluster_frame) and \
                        self.data_selection in self.cluster_assignments_dict.keys() and self.cluster_type == 'mask':
                    annot_mask = generate_mask_with_cluster_annotations(self.mask_config[self.mask_selection]["raw"],
                                self.cluster_frame[self.data_selection],
                                self.cluster_assignments_dict[self.data_selection], use_gating_subset=self.apply_gating,
                                                                        gating_subset_list=self.gating_cell_id_list)
                    annot_mask = annot_mask if annot_mask is not None else \
                        np.where(self.mask_config[self.mask_selection]["array"].astype(np.uint8) > 0, 255, 0)
                    image = cv2.addWeighted(image.astype(np.uint8), 1, annot_mask, mask_level, 0)
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
                                # TODO: decide if compression level should be toggleable
                                # currently set to lowest possible compression level for speed
                                binary_compression_level=1)

    def overlay_grid_on_additive_image(self, image: Union[np.array, np.ndarray]):
        if self.overlay_grid:
            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    generate_greyscale_grid_array((image.shape[0],
                                    image.shape[1])).astype(np.uint8), 1, 0)
        return image

    def apply_gating_to_canvas_mask_image(self, mask: Union[np.array, np.ndarray]):
        if self.apply_gating:
            mask = subset_mask_outline_using_cell_id_list(self.mask_config[self.mask_selection]["raw"],
                                                          self.mask_config[self.mask_selection]["raw"],
                                                          self.gating_cell_id_list).astype(np.uint8)
        return mask

    def overlay_mask_outline_on_mask_image(self, image: Union[np.array, np.ndarray]):
        if self.add_mask_boundary and self.mask_config[self.mask_selection]["boundary"] is not None:
            # add the border of the mask after converting back to greyscale to derive the conversion
            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    self.mask_config[self.mask_selection]["boundary"].astype(np.uint8), 1, 0)
        return image

    def generate_canvas(self) -> Union[go.Figure, dict]:
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
            try:
                fig = self.transfer_canvas_data_to_existing_canvas()
                # del cur_graph
            # key error could happen if the canvas is reset with no layers, so rebuild from scratch
            except (KeyError, TypeError, ValueError):
                fig = self.canvas
                fig['layout']['uirevision'] = True

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
            fig['layout']['uirevision'] = True

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
        # fig = canvas_layers[image_type][currently_selected[0]]
        fig = self.add_canvas_legend_text(fig, x_axis_placement)

        # set the x-axis scale placement based on the size of the image
        # for adding a scale bar
        fig = self.add_canvas_scalebar(fig, x_axis_placement)

        fig = self.add_canvas_hover_template(fig)
        return fig.to_dict()

    def get_shape(self):
        return self.image.shape

    def get_image(self):
        return self.image

    def current_canvas_exists(self, hover_template_exists: bool=False):
        """
        Return if the current canvas passed is a real canvas object
        """
        return 'layout' in self.cur_graph and 'uirevision' in self.cur_graph['layout'] and \
                self.cur_graph['layout']['uirevision'] and not hover_template_exists

    def transfer_canvas_data_to_existing_canvas(self):
        """
        Transfer the newly created canvas image to the imported canvas if it has existing parameters such as
        uirevision
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

    def add_canvas_legend_text(self, fig: go.Figure, x_axis_placement: Union[int, float]):
        """
        Add canvas legend text using a specified text size and x axis placement
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary
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

    def add_canvas_hover_template(self, fig: go.Figure):
        # the masking mask ID get priority over the channel intensity hover
        # TODO: combine both the mask ID and channel intensity into one hover if both are requested

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
            # fig.update_traces(hovertemplate=new_hover)
        else:
            fig.update(data=[{'customdata': None}])
            new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"
        fig.update_traces(hovertemplate=new_hover)
        return fig


    def add_canvas_scalebar(self, fig: go.Figure, x_axis_placement: Union[int, float]):
        """
        Add a canvas scalebar with a set bar width of 2 and a number size set by the user
        The y coordinate is always fixed at 0.05 to make it readily identifiable when parsing the shape dictionary
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
    def set_default_canvas_layout(fig: go.Figure):
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis=XAxis(showticklabels=False, domain=[0, 1]),
                          yaxis=YAxis(showticklabels=False),
                          margin=dict(
                              l=10,
                              r=0,
                              b=25,
                              t=35,
                              pad=0))
        return fig


class CanvasLayout:
    """
    This class represents a set of layout manipulations for the image canvas. It is distinct from
    CanvasImage in that it doesn't manipulate the underlying image, but rather the layout and UI components
    of the attributes projected on top of the data. It expected a `go.Figure` object or a dictionary representing
    a `go.Figure` object as input
    """
    def __init__(self, figure: Union[dict, go.Figure]):
        if 'layout' in figure and 'shapes' in figure['layout'] and \
                len(figure['layout']['shapes']) > 0 and not \
                isinstance(figure['layout']['shapes'], tuple):
            figure['layout']['shapes'] = [shape for shape in figure['layout']['shapes'] if \
                                          shape and not is_bad_shape(shape)]
        try:
            figure['layout']['yaxis']['domain'] = [0, 1]
            figure['layout']['xaxis']['domain'] = [0, 1]
        except KeyError:
            pass
        self.figure = figure
        # TODO: add condition checking whether the annotations or shapes are held in tuples (do not allow)
        if 'layout' in self.figure and 'annotations' in self.figure['layout'] and \
                len(self.figure['layout']['annotations']) > 0 and not \
                isinstance(self.figure['layout']['annotations'], tuple):
            self.cur_annotations = [annot for annot in self.figure['layout']['annotations'] if annot is not None]
        else:
            self.cur_annotations = []
        if 'layout' in self.figure and 'shapes' in self.figure['layout'] and \
                len(self.figure['layout']['shapes']) > 0 and not \
                isinstance(self.figure['layout']['shapes'], tuple):
            self.cur_shapes = [shape for shape in self.figure['layout']['shapes'] if shape and \
                               'type' in shape and not is_bad_shape(shape)]
        else:
            self.cur_shapes = []

        for shape in self.cur_shapes:
            if 'label' in shape and 'texttemplate' in shape['label']:
                shape['label'] = {}

    def get_fig(self):
        self.figure['layout']['shapes'] = self.cur_shapes
        self.figure['layout']['annotations'] = self.cur_annotations
        return self.figure

    def add_scalebar(self, x_axis_placement, invert_annot, pixel_ratio, image_shape, legend_size,
                     proportion=0.1, annotation_color: str="white"):
        try:
            proportion = float(proportion / pixel_ratio)
        except ZeroDivisionError:
            pass
        fig = go.Figure(self.figure)
        fig.update_layout(newshape=dict(line=dict(color="white")))
        # TODO: request for custom scalebar value to change the length of the bar, can implement here
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
            if not x_range_high >= x_range_low: raise AssertionError
            custom_scale_val = int(float(math.ceil(int(proportion *
                                (x_range_high - x_range_low))) + 1) * float(pixel_ratio))
        except (KeyError, TypeError, AssertionError):
            custom_scale_val = None

        fig = add_scale_value_to_figure(fig, image_shape, scale_value=custom_scale_val,
                                font_size=legend_size, x_axis_left=x_axis_placement, invert=invert_annot,
                                proportion=proportion, scale_color=annotation_color)

        return fig.to_dict()

    def add_legend_text(self, legend_text, x_axis_placement, legend_size):
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
    def toggle_legend(self, toggle_legend: bool, legend_text, x_axis_placement, legend_size):
        """
        Modify the legend text for the figure, or remove the legend
        """
        cur_annotations = [annot for annot in self.cur_annotations if \
                           annot is not None and 'y' in annot and annot['y'] != 0.05]
        self.figure['layout']['annotations'] = cur_annotations
        if not toggle_legend:
            return self.figure
        else:
            return self.add_legend_text(legend_text, x_axis_placement, legend_size)

    def toggle_scalebar(self, toggle_scalebar, x_axis_placement, invert_annot,
                        pixel_ratio, image_shape, legend_size, proportion=0.1, scalebar_color: str="white"):
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
        if not toggle_scalebar:
            return self.figure
        else:
            return self.add_scalebar(x_axis_placement, invert_annot,
                    pixel_ratio, image_shape, legend_size, proportion, scalebar_color)

    def change_annotation_size(self, legend_size):
        """
        Change the size of the legend and scalebar
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

    def add_point_annotations_as_circles(self, imported_annotations, cur_image, circle_size):
        """
        Add a circle for each point annotation in a CSV file. Each annotation is validated against the
        image dimensions in the current canvas to ensure that the annotation lies within the dimensions
        """
        imported_annotations = pd.DataFrame(imported_annotations)
        # fig = go.Figure(self.figure)
        # TODO: figure out what to increase the speed of shape rendering
        for index, row in imported_annotations.iterrows():
            if validate_coordinate_set_for_image(row['x'], row['y'], cur_image):
                self.cur_shapes.append(
                    {'editable': True, 'line': {'color': 'white'}, 'type': 'circle',
                     'x0': (row['x'] - circle_size), 'x1': (row['x'] + circle_size),
                     'xref': 'x', 'y0': (row['y'] - circle_size), 'y1': (row['y'] + circle_size), 'yref': 'y'})
                # fig.add_shape(type="circle",
                #               xref='x', yref='y',
                #               x0=(row['x'] - circle_size), y0=(row['y'] - circle_size), x1=(row['x'] + circle_size),
                #               y1=(row['y'] + circle_size),
                #               line_color="white", editable=True)
        self.figure['layout']['shapes'] = self.cur_shapes
        return self.figure

    def update_scalebar_zoom_value(self, current_graph_layout, pixel_ratio, proportion=0.1,
                                   scalebar_col: str="white"):
        """
        update the scalebar value when zoom is used
        Loop through the annotations to identify the scalebar value when y = 0.06
        """
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
                if 'xaxis.range[0]' and 'xaxis.range[1]' in current_graph_layout:
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
                if not x_range_high >= x_range_low: raise AssertionError
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

    def use_custom_scalebar_value(self, custom_scale_val, pixel_ratio, proportion=0.1):
        # self.figure = strip_invalid_shapes_from_graph_layout(self.figure)
        pixel_ratio = pixel_ratio if pixel_ratio is not None and pixel_ratio > 0 else 1
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

    def clear_improper_shapes(self):

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
                                           gating_cell_id_list: list=None):
        """
        Add an annotation circle to every mask object in a mask, or in a list of gated objects
        requires:
        mask = a mask with raw object values starting at 1 in numpy int32 form
        cluster_frame = a dataframe with the columns `cell_id` and `cluster`
        cluster_assignments = a dictionary of cluster labels corresponding to a hex colour
        data_selection = string representation of the current ROI
        """
        shapes = self.cur_shapes if self.cur_shapes else []
        cluster_frame = pd.DataFrame(cluster_frame)
        cluster_frame = cluster_frame.astype(str)
        ids_use = gating_cell_id_list if (gating_cell_id_list is not None and use_gating) else np.unique(mask)
        for mask_id in ids_use:
            # IMP: each region needs to be subset before region props are computed, or the centroids are wrong
            subset = np.where(mask == int(mask_id), int(mask_id), 0)
            region_props = measure.regionprops(subset)
            for region in region_props:
                center = region.centroid
            annotation = pd.Series(cluster_frame[cluster_frame['cell_id'] == str(mask_id)]['cluster']).to_list()
            if annotation:
                annotation = str(annotation[0])
                # boundary[int(center[0]), int(center[1])] = mask_id
                shapes.append(
                    {'editable': False, 'line': {'color': 'white'}, 'type': 'circle',
                     'x0': (int(center[1]) - circle_size), 'x1': (int(center[1]) + circle_size),
                     'xref': 'x', 'y0': (int(center[0]) - circle_size), 'y1': (int(center[0]) + circle_size),
                     'yref': 'y',
                     'fillcolor': cluster_assignments[data_selection][annotation]})
        self.figure['layout']['shapes'] = shapes
        return self.figure

    def remove_cluster_annotation_shapes(self):
        """
        Remove the cluster annotation shapes from the canvas.
        These are uniquely recognized as circles that are not editable
        """
        new_shapes = []
        for shape in self.cur_shapes:
            if 'editable' not in shape or not is_cluster_annotation_circle(shape):
                new_shapes.append(shape)
        self.figure['layout']['shapes'] = new_shapes
        return self.figure

    def update_coordinate_window(self, current_window, x_request, y_request):
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

def reset_graph_with_malformed_template(graph: Union[go.Figure, dict]):
    """
    Parse a current graph that may have malformed shapes (i.e. a shape with a blank texttemplate in the 'label'
    slot), and return a cleaned graph dictionary object with the dragmode set to zoom
    """
    graph = graph.to_dict() if not isinstance(graph, dict) else graph
    fig = go.Figure(CanvasLayout(graph).get_fig())
    fig.update_layout(dragmode="zoom")
    return fig
