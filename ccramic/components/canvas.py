import numpy as np
import plotly.graph_objs as go
import cv2
import plotly.express as px
from PIL import Image
from ccramic.utils.cell_level_utils import generate_greyscale_grid_array
from ccramic.inputs.pixel_level_inputs import add_scale_value_to_figure
from ccramic.utils.pixel_level_utils import per_channel_intensity_hovertext
from plotly.graph_objs.layout import YAxis, XAxis

class CanvasImage:
    """
    This class generates a canvas `go.Figure` with the current selected channels and various
    UI configurations
    """
    def __init__(self, canvas_layers: dict, data_selection: str, currently_selected: list,
                 mask_config: dict, mask_selection: str, mask_blending_level: int,
                 overlay_grid: list, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                 legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                 show_each_channel_intensity, raw_data_dict, aliases):
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
        self.cur_graph = cur_graph
        self.pixel_ratio = pixel_ratio if pixel_ratio is not None else 1
        self.legend_text = legend_text
        self.toggle_scalebar = toggle_scalebar
        self.legend_size = legend_size
        self.toggle_legend = toggle_legend
        self.add_cell_id_hover = add_cell_id_hover
        self.show_each_channel_intensity = show_each_channel_intensity
        self.raw_data_dict = raw_data_dict
        self.aliases = aliases

        image = sum([self.canvas_layers[self.data_selection][elem].astype(np.float32) for \
                     elem in self.currently_selected if \
                     elem in self.canvas_layers[self.data_selection].keys()]).astype(np.float32)
        self.image = np.clip(image, 0, 255)

        if self.mask_toggle and None not in (self.mask_config, self.mask_selection) and len(self.mask_config) > 0:
            if self.image.shape[0] == self.mask_config[self.mask_selection]["array"].shape[0] and \
                    self.image.shape[1] == self.mask_config[self.mask_selection]["array"].shape[1]:
                # set the mask blending level based on the slider, by default use an equal blend
                mask_level = float(self.mask_blending_level / 100) if self.mask_blending_level is not None else 1
                image = cv2.addWeighted(self.image.astype(np.uint8), 1,
                                        self.mask_config[self.mask_selection]["array"].astype(np.uint8), mask_level, 0)
                if self.add_mask_boundary and self.mask_config[self.mask_selection]["boundary"] is not None:
                    # add the border of the mask after converting back to greyscale to derive the conversion
                    image = cv2.addWeighted(image.astype(np.uint8), 1,
                                            self.mask_config[self.mask_selection]["boundary"].astype(np.uint8), 1, 0)

        if ' overlay grid' in self.overlay_grid:
            image = cv2.addWeighted(image.astype(np.uint8), 1,
                                    generate_greyscale_grid_array((image.shape[0], image.shape[1])), 1, 0)

        self.canvas = px.imshow(Image.fromarray(image.astype(np.uint8)))
        # fig.update(data=[{'customdata':)
    def generate_canvas(self) -> go.Figure:
        fig = self.canvas.update_traces(hoverinfo="skip")
        x_axis_placement = 0.00001 * self.image.shape[1]
        # make sure the placement is min 0.05 and max 0.1
        x_axis_placement = x_axis_placement if 0.05 <= x_axis_placement <= 0.15 else 0.05
        if self.invert_annot:
            x_axis_placement = 1 - x_axis_placement
        # if the current graph already has an image, take the existing layout and apply it to the new figure
        # otherwise, set the uirevision for the first time
        # fig = add_scale_value_to_figure(fig, image_shape, x_axis_placement)
        # do not update if there is already a hover template as it will be too slow
        # scalebar is y = 0.06
        # legend is y = 0.05
        hover_template_exists = 'data' in self.cur_graph and 'customdata' in self.cur_graph['data'] and \
                                self.cur_graph['data']['customdata'] is not None
        if 'layout' in self.cur_graph and 'uirevision' in self.cur_graph['layout'] and \
                self.cur_graph['layout']['uirevision'] and not hover_template_exists:
            try:
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
                fig = self.cur_graph
                # del cur_graph
            # keyerror could happen if the canvas is reset with no layers, so rebuild from scratch
            except (KeyError, TypeError):
                fig = self.canvas
                fig['layout']['uirevision'] = True

                if self.toggle_scalebar:
                    fig = add_scale_value_to_figure(fig, self.get_shape(), font_size=self.legend_size,
                                                    x_axis_left=x_axis_placement, pixel_ratio=self.pixel_ratio,
                                                    invert=self.invert_annot)

                fig = go.Figure(fig)
                fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                                  xaxis=XAxis(showticklabels=False, domain=[0, 1]),
                                  yaxis=YAxis(showticklabels=False),
                                  margin=dict(
                                      l=10,
                                      r=0,
                                      b=25,
                                      t=35,
                                      pad=0
                                  ))
                fig.update_layout(hovermode="x")
        else:
            fig = self.canvas
            # del cur_graph
            # if making the fig for the first time, set the uirevision
            fig['layout']['uirevision'] = True

            if self.toggle_scalebar:
                fig = add_scale_value_to_figure(fig, self.get_shape(), font_size=self.legend_size,
                                                x_axis_left=x_axis_placement, pixel_ratio=self.pixel_ratio,
                                                invert=self.invert_annot)

            fig = go.Figure(fig)
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                              xaxis=XAxis(showticklabels=False),
                              yaxis=YAxis(showticklabels=False),
                              margin=dict(
                                  l=10,
                                  r=0,
                                  b=25,
                                  t=35,
                                  pad=0
                              ))
            fig.update_layout(hovermode="x")

        fig = go.Figure(fig)
        fig.update_layout(newshape=dict(line=dict(color="white")))

        # set how far in from the lefthand corner the scale bar and colour legends should be
        # higher values mean closer to the centre
        # fig = canvas_layers[image_type][currently_selected[0]]
        if self.legend_text != '' and self.toggle_legend:
            fig.add_annotation(text=self.legend_text, font={"size": self.legend_size + 1}, xref='paper',
                               yref='paper',
                               x=(1 - x_axis_placement),
                               # xanchor='right',
                               y=0.05,
                               # yanchor='bottom',
                               bgcolor="black",
                               showarrow=False)

        # set the x-axis scale placement based on the size of the image
        # for adding a scale bar
        if self.toggle_scalebar:
            # set the x0 and x1 depending on if the bar is inverted or not
            x_0 = x_axis_placement if not self.invert_annot else (x_axis_placement - 0.075)
            x_1 = (x_axis_placement + 0.075) if not self.invert_annot else x_axis_placement
            fig.add_shape(type="line",
                          xref="paper", yref="paper",
                          x0=x_0, y0=0.05, x1=x_1,
                          y1=0.05, line=dict(color="white", width=2))

        # set the custom hovertext if is is requested
        # the masking mask ID get priority over the channel intensity hover
        # TODO: combine both the mask ID and channel intensity into one hover if both are requested

        if self.mask_toggle and None not in (self.mask_config, self.mask_selection) and len(self.mask_config) > 0 and \
                ' show mask ID on hover' in self.add_cell_id_hover:
            try:
                # fig.update(data=[{'customdata': None}])
                fig.update(data=[{'customdata': self.mask_config[self.mask_selection]["hover"]}])
                new_hover = per_channel_intensity_hovertext(["mask ID"])
            except KeyError:
                new_hover = "x: %{x}<br>y: %{y}<br><extra></extra>"

        elif " show channel intensities on hover" in self.show_each_channel_intensity:
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

    def get_shape(self):
        return self.image.shape

    def get_image(self):
        return self.image
