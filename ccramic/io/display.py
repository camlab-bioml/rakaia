from numpy.core._exceptions import _ArrayMemoryError
import pandas as pd
from ccramic.utils.region import RectangleRegion, FreeFormRegion
import os
from tifffile import imwrite
import numpy as np
import plotly.graph_objs as go
from typing import Union
from ccramic.inputs.pixel_level_inputs import set_roi_identifier_from_length

class RegionSummary:
    """
    Produces a dictionary or dataframe of one or more regions of one of more channels
    with summary pixel-level statistics for minimum, mean, and maximum array values
    Statistics may be computed for various types of shapes drawn on the interactive canvas,
    and identified from the `dash`-based graph layout: zoom, rectangle shapes, and svg freeform paths
    """
    def __init__(self, graph_layout, image_dict, layers, data_selection, aliases_dict):
        self.graph_layout = graph_layout
        self.image_dict = image_dict
        self.data_selection = data_selection
        self.aliases = aliases_dict
        # these are the zoom keys by default
        self.zoom_keys  = ('xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]')
        # if a rectangle is modified in the canvas, these are the new keys in the layout dictionary
        self.modified_rect_keys = ('shapes[1].x0', 'shapes[1].x1', 'shapes[1].y0', 'shapes[1].y1')
        self.selected_channels = layers
        # initialize the empty frame
        self.summary_frame = pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                             'Min': [], 'Total': []}).to_dict(orient='records')

        if 'shapes' in self.graph_layout and len(self.graph_layout['shapes']) > 0:
            self.compute_statistics_shapes()
        elif ('shapes' not in self.graph_layout or len(self.graph_layout['shapes']) <= 0) and \
                all([elem in self.graph_layout for elem in self.zoom_keys]):
            self.compute_statistics_rectangle(reg_type="zoom", redrawn=False)
        elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                all([elem in graph_layout for elem in self.modified_rect_keys]):
            self.compute_statistics_rectangle(reg_type="rect", redrawn=True)
        elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
                all(['shapes' in elem and 'path' in elem for elem in graph_layout.keys()]):
            self.compute_statistics_modified_svg_path()

    def compute_statistics_shapes(self):
        """
        Compute the region statistics when new shapes are drawn on the canvas and are not modified
        """
        # these are for each sample
        mean_panel = []
        max_panel = []
        min_panel = []
        total_panel = []
        aliases = []
        region = []
        region_index = 1
        shapes_keep = [shape for shape in self.graph_layout['shapes'] if shape['type'] not in ['line']]
        for shape in shapes_keep:
            try:
                # option 1: if the shape is drawn with a rectangle
                if shape['type'] == 'rect':
                    for layer in self.selected_channels:
                        region_shape = RectangleRegion(self.image_dict[self.data_selection][layer],
                                                       shape, reg_type="rect")
                        mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                        max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                        min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                        total_panel.append(round(float(region_shape.compute_integrated_signal()), 2))
                        aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)
                        region.append(region_index)
                    # option 2: if a closed form shape is drawn
                elif shape['type'] == 'path' and 'path' in shape:
                    for layer in self.selected_channels:
                        region_shape = FreeFormRegion(self.image_dict[self.data_selection][layer], shape)
                        mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                        max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                        min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                        total_panel.append(round(float(region_shape.compute_integrated_signal()), 2))
                        aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)
                        region.append(region_index)
                region_index += 1
                # mean_panel.append(round(sum(shapes_mean) / len(shapes_mean), 2))
                # max_panel.append(round(sum(shapes_max) / len(shapes_max), 2))
                # min_panel.append(round(sum(shapes_min) / len(shapes_min), 2))

            # TODO: evaluate if this exception catch actually works, since there is already one in the region child classes
            except (AssertionError, ValueError, ZeroDivisionError, IndexError, TypeError,
                    _ArrayMemoryError, KeyError):
                pass

        layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel,
                      'Total': total_panel, 'Region': region}
        self.summary_frame = pd.DataFrame(layer_dict).to_dict(orient='records')

    def compute_statistics_rectangle(self, reg_type="zoom", redrawn=False):
        """
        Compute the region statistics when a rectangular shape or zoom is enabled
        `reg_type` will specify the type of shape or zoom used for the region,
        and `redrawn` specifies if the keys are modified when an existing shape is changed
        """
        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            total_panel = []
            aliases = []

            for layer in self.selected_channels:
                region = RectangleRegion(self.image_dict[self.data_selection][layer],
                                         self.graph_layout, reg_type=reg_type, redrawn=redrawn)
                mean_panel.append(round(float(region.compute_pixel_mean()), 2))
                max_panel.append(round(float(region.compute_pixel_max()), 2))
                min_panel.append(round(float(region.compute_pixel_min()), 2))
                total_panel.append(round(float(region.compute_integrated_signal()), 2))
                aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel,
                          'Total': total_panel}
            self.summary_frame = pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
            pass

    def compute_statistics_modified_svg_path(self):
        """
        Compute the region statistics when a freeform svg path is drawn, then modified on the canvas
        """
        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            total_panel = []
            aliases = []
            for layer in self.selected_channels:
                for shape_path in self.graph_layout.values():
                    region_shape = FreeFormRegion(self.image_dict[self.data_selection][layer], shape_path)
                    mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                    max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                    min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                    total_panel.append(round(float(region_shape.compute_integrated_signal()), 2))
                aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel,
                          'Total': total_panel}

            self.summary_frame = pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError, AttributeError):
            pass

    def get_summary_frame(self):
        return self.summary_frame


def output_current_canvas_as_tiff(canvas_image, dest_dir="/tmp/", output_default="canvas",
                                  roi_name: str=None, use_roi_name=False, delimiter:str="+++"):
    """
    Output the current canvas image as a photometric tiff
    """
    if canvas_image is not None:
        outname = set_roi_identifier_from_length(roi_name, delimiter=delimiter) if use_roi_name else output_default
        dest_file = str(os.path.join(dest_dir, f"{outname}.tiff"))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        imwrite(dest_file, canvas_image.astype(np.uint8), photometric='rgb')
        return dest_file
    else:
        return None

def output_current_canvas_as_html(cur_graph, canvas_style, dest_dir=None, roi_name: str=None, delimiter: str="+++",
                                  use_roi_name=False, output_default:str="canvas"):
    """
    Output the current `dcc.Graph` object as HTML with the same aspect ratio as the underlying array
    Returns the filepath string for `dcc.send_file`
    """
    fig = go.Figure(cur_graph)
    # fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
    #                   xaxis=XAxis(showticklabels=False),
    #                   yaxis=YAxis(showticklabels=False),
    #                   margin=dict(l=0, r=0, b=0, t=0, pad=0))
    fig.update_layout(dragmode="zoom")
    outname = set_roi_identifier_from_length(roi_name, delimiter=delimiter) if use_roi_name else output_default
    outfile = str(os.path.join(dest_dir, f"{outname}.html"))
    fig.write_html(outfile, default_width=canvas_style['width'],
                   default_height=canvas_style['height'])
    return outfile


class FullScreenCanvas:
    """
    Represents a `go.Figure` instance of a blended canvas with the annotations and shapes
    removed
    """
    def __init__(self, canvas: Union[dict, go.Figure], canvas_layout: dict):
        self.canvas = canvas
        self.canvas_layout = canvas_layout
        if 'layout' in self.canvas_layout.keys() and 'annotations' in self.canvas_layout['layout'].keys() and \
                len(self.canvas_layout['layout']['annotations']) > 0:
            self.canvas_layout['layout']['annotations'] = []
        if 'shapes' in canvas_layout.keys():
            self.canvas_layout['shapes'] = {}
        if 'layout' in self.canvas.keys() and 'annotations' in self.canvas['layout'].keys() and \
                len(self.canvas['layout']['annotations']) > 0:
            self.canvas['layout']['annotations'] = []
        if 'layout' in self.canvas.keys() and 'shapes' in self.canvas['layout'].keys():
            self.canvas['layout']['shapes'] = []

    def get_canvas(self):
        return self.canvas

    def get_canvas_layout(self):
        return self.canvas_layout
