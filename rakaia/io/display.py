import os
import datetime
from typing import Union
from numpy.core._exceptions import _ArrayMemoryError
import pandas as pd
from tifffile import imwrite
import numpy as np
import plotly.graph_objs as go
from dash import html
from plotly.graph_objs import XAxis, YAxis
from rakaia.utils.region import (
    RectangleRegion,
    FreeFormRegion,
    AnnotationPreviewGenerator,
    RegionStatisticGroups)
from rakaia.inputs.pixel import set_roi_identifier_from_length
from rakaia.components.canvas import CanvasLayout

def empty_region_table():
    """
    Generate a dictionary representation of an empty pandas Dataframe for channel region summaries
    """
    layers = {}
    for key in list(RegionStatisticGroups.columns):
        layers[key] = []
    return pd.DataFrame(layers).to_dict(orient='records')

class RegionSummary:
    """
    Produces a dictionary or dataframe of one or more regions of one of more channels
    with summary pixel-level statistics for minimum, mean, and maximum array values
    Statistics may be computed for various types of shapes drawn on the interactive canvas,
    and identified from the `dash`-based graph layout: zoom, rectangle shapes, and svg freeform paths

    :param graph: Current `go.Figure` canvas object
    :param graph_layout: Dictionary of canvas layout modifications
    :param image_dict: Dictionary of raw channel intensity numpy arrays
    :param layers: List of currently selected channels in the blend canvas
    :param data_selection: String representation of the current ROI selection
    :param aliases_dict: Dictionary matching channel ids to display values
    :return: None
    """
    def __init__(self, graph: Union[go.Figure, dict], graph_layout: dict, image_dict: dict,
                 layers: Union[list, dict], data_selection: str, aliases_dict: dict):
        self.graph = graph
        # by default, use the shapes or coordinates in the layout, but override below if not zoom
        self.graph_layout = graph_layout
        self.image_dict = image_dict
        self.data_selection = data_selection
        self.aliases = aliases_dict
        # these are the zoom keys by default
        self.zoom_keys = ('xaxis.range[1]', 'xaxis.range[0]', 'yaxis.range[1]', 'yaxis.range[0]')
        self.selected_channels = layers
        self.summary_frame = empty_region_table()
        # if zoom is used
        if ('shapes' not in self.graph_layout or len(self.graph_layout['shapes']) <= 0) and \
             all(elem in self.graph_layout for elem in self.zoom_keys):
            self.compute_statistics_rectangle(reg_type="zoom", redrawn=False)
        # if zoom isn't used, then iterate the shapes from the graph component instead of the layout.
        # Allows to get all shapes even if one was edited
        else:
            if 'layout' in self.graph and 'shapes' in self.graph['layout'] and self.graph['layout']['shapes']:
                self.graph_layout = self.graph['layout']
                self.compute_statistics_shapes()

    def define_shapes_to_keep(self):
        """

        :return: list of shapes that should be retained to annotate (must not be lines, and must be editable)
        """
        shapes_keep = []
        if 'shapes' in self.graph_layout:
            for shape in self.graph_layout['shapes']:
                try:
                    if 'type' in shape and shape['type'] not in ['line'] and 'editable' in shape and shape['editable']:
                        shapes_keep.append(shape)
                except KeyError:
                    pass
        return shapes_keep

    def compute_statistics_shapes(self):
        """
        Compute the region statistics when new shapes are drawn on the canvas and are not modified

        :return: None
        """
        # these are for each sample
        mean_panel = []
        max_panel = []
        min_panel = []
        median_panel = []
        std_panel = []
        total_panel = []
        aliases = []
        region = []
        region_index = 1
        shapes_keep = self.define_shapes_to_keep()
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
                        median_panel.append(round(float(region_shape.compute_pixel_median()), 2))
                        std_panel.append(round(float(region_shape.compute_pixel_dev()), 2))
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
                        median_panel.append(round(float(region_shape.compute_pixel_median()), 2))
                        std_panel.append(round(float(region_shape.compute_pixel_dev()), 2))
                        total_panel.append(round(float(region_shape.compute_integrated_signal()), 2))
                        aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)
                        region.append(region_index)
                region_index += 1

            # exception may not be necessary as there are already catches in the child classes
            except (AssertionError, ValueError, ZeroDivisionError, IndexError, TypeError,
                    _ArrayMemoryError, KeyError):
                pass

        layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel,
                      'Median': median_panel, 'SD': std_panel, 'Total': total_panel, 'Region': region}
        self.summary_frame = pd.DataFrame(layer_dict).to_dict(orient='records')


    def compute_statistics_rectangle(self, reg_type="zoom", redrawn=False):
        """
        Compute the region statistics when a rectangular shape or zoom is enabled
        `reg_type` will specify the type of shape or zoom used for the region,
        and `redrawn` specifies if the keys are modified when an existing shape is changed

        :return: None
        """
        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            median_panel = []
            std_panel = []
            total_panel = []
            aliases = []

            for layer in self.selected_channels:
                region = RectangleRegion(self.image_dict[self.data_selection][layer],
                                         self.graph_layout, reg_type=reg_type, redrawn=redrawn)
                mean_panel.append(round(float(region.compute_pixel_mean()), 2))
                max_panel.append(round(float(region.compute_pixel_max()), 2))
                min_panel.append(round(float(region.compute_pixel_min()), 2))
                median_panel.append(round(float(region.compute_pixel_median()), 2))
                std_panel.append(round(float(region.compute_pixel_dev()), 2))
                total_panel.append(round(float(region.compute_integrated_signal()), 2))
                aliases.append(self.aliases[layer] if layer in self.aliases.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel,
                          'Median': median_panel, 'SD': std_panel, 'Total': total_panel}
            self.summary_frame = pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
            pass

    def get_summary_frame(self):
        """
        Get the summary frame for all regions evaluated

        :return: `pd.DataFrame` of summarized regions for all currently selected channels
        """
        return self.summary_frame


def output_current_canvas_as_tiff(canvas_image, dest_dir: str=None, output_default="canvas",
                                  roi_name: str=None, use_roi_name=False, delimiter: str="+++"):
    """
    Output the current canvas image as a photometric tiff
    """
    if canvas_image is not None:
        out_tiff_name = set_roi_identifier_from_length(roi_name, delimiter=delimiter) if use_roi_name else output_default
        dest_file = str(os.path.join(dest_dir, f"{out_tiff_name}.tiff"))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        imwrite(dest_file, canvas_image.astype(np.uint8), photometric='rgb')
        return dest_file
    return None

def output_current_canvas_as_html(dest_dir=None, cur_graph=None, canvas_style=None,
                                roi_name: str=None, delimiter: str="+++",
                                use_roi_name=False, output_default:str="canvas"):
    """
    Output the current `dcc.Graph` object as HTML with the same aspect ratio as the underlying array
    Returns the filepath string for `dcc.send_file`
    """
    # ensure that the domains are between 0 and 1
    if dest_dir and cur_graph and canvas_style:
        try:
            cur_graph['layout']['yaxis']['domain'] = [0, 1]
            cur_graph['layout']['xaxis']['domain'] = [0, 1]
        except KeyError:
            pass
        cur_graph = CanvasLayout(cur_graph)
        fig = go.Figure(cur_graph.get_fig())
        fig.update_layout(dragmode="zoom")
        out_name = set_roi_identifier_from_length(roi_name, delimiter=delimiter) if use_roi_name else output_default
        outfile = str(os.path.join(dest_dir, f"{out_name}.html"))
        fig.write_html(outfile, default_width=canvas_style['width'],
                       default_height=canvas_style['height'])
        return outfile
    return None


class FullScreenCanvas:
    """
    Represents a `go.Figure` instance of a blended canvas with the annotations and shapes removed

    :param canvas: Current canvas  `go.Figure` object
    :param canvas_layout: Dictionary of canvas layout modifications

    :return: None
    """
    def __init__(self, canvas: Union[dict, go.Figure], canvas_layout: dict):
        self.canvas = canvas
        self.canvas_layout = canvas_layout
        if 'layout' in self.canvas.keys():
            self.clear_canvas_annotation()
            self.clear_canvas_shapes()
        self.clear_layout_annotations()
        self.clear_layout_shapes()

    def clear_canvas_annotation(self):
        """
        Remove the canvas annotations for full screen mode

        :return: None
        """
        if 'annotations' in self.canvas['layout'].keys() and \
                len(self.canvas['layout']['annotations']) > 0:
            self.canvas['layout']['annotations'] = []

    def clear_canvas_shapes(self):
        """
        Remove the canvas shapes for full screen mode

        :return: None
        """
        if 'shapes' in self.canvas['layout'].keys():
            self.canvas['layout']['shapes'] = []

    def clear_layout_annotations(self):
        """
        Remove the annotations for the canvas layout object for full screen mode

        :return: None
        """
        if 'layout' in self.canvas_layout.keys() and 'annotations' in self.canvas_layout['layout'].keys() and \
                len(self.canvas_layout['layout']['annotations']) > 0:
            self.canvas_layout['layout']['annotations'] = []

    def clear_layout_shapes(self):
        """
        Remove the shapes for the canvas layout object for full screen mode

        :return: None
        """
        if 'shapes' in self.canvas_layout.keys():
            self.canvas_layout['shapes'] = {}


    def get_canvas(self, as_fig: bool=False) -> Union[go.Figure, dict]:

        """
        Get the modified canvas

        :param as_fig: Whether the canvas should be returned as a `go.Figure` object, or as a dictionary (default).

        :return: `go.Figure` or dictionary representation of the fullscreen canvas object
        """
        if as_fig:
            fig = go.Figure(self.canvas)
            fig.update_layout(dragmode='pan')
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis=XAxis(showticklabels=False),
                              yaxis=YAxis(showticklabels=False), margin=dict(l=0, r=0, b=0, t=0, pad=0))
            return fig
        return self.canvas

    def get_canvas_layout(self):
        return self.canvas_layout

def preset_options_preview_text(preset_dict: dict=None):
    """
    Generate the HTML compatible text that supplies the list of possible presets that the user can select
    """
    text = ''
    if preset_dict:
        for stud, val in preset_dict.items():
            try:
                try:
                    low_bound = round(float(val['x_lower_bound']))
                except TypeError:
                    low_bound = None
                try:
                    up_bound = round(float(val['x_upper_bound']))
                except TypeError:
                    up_bound = None
                text = text + f"{stud}: \r\n l_bound: {low_bound}, " \
                          f"y_bound: {up_bound}, filter type: {val['filter_type']}, " \
                          f"filter val: {val['filter_val']}, filter_sigma: {val['filter_sigma']} \r\n"
            except KeyError:
                pass
    return text


def annotation_preview_table(annotation_dict: dict=None, roi_selection: str=None):
    """
    Generate a preview table for the annotations in the current ROI
    The preview table is returned as a list of dictionaries, with each entry corresponding to one annotation
    Columns are returned as a list od dictionaries, with each entry corresponding to one HTML-compatible column
    in the dash data frame
    """
    if annotation_dict and roi_selection and roi_selection in annotation_dict.keys():
        columns_keep = ['id', 'cell_type', 'annotation_column', 'type']
        if len(annotation_dict[roi_selection]) > 0:
            preview_writer = AnnotationPreviewGenerator()
            annotation_list = []
            for key, value in annotation_dict[roi_selection].items():
                try:
                    value = dict((k, value[k]) for k in columns_keep)
                    for sub_key, sub_value in value.items():
                        value[sub_key] = str(sub_value)
                    value['preview'] = preview_writer.annotation_preview(key, value['type'])
                    annotation_list.append(value)
                except (KeyError, ValueError, TypeError):
                    pass
            columns = [{'id': p, 'name': p, 'editable': False} for p in columns_keep + ["preview"]]
            return annotation_list, columns
    return [], []

def timestamp_download_child(download_type: str="Canvas (tiff)"):
    """
    Generate an HTML compatible alert when a session download is completed
    with a download type passed as a string, and the current timestamp
    """
    if download_type:
        return html.H6(f'{download_type} downloaded: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    return []
