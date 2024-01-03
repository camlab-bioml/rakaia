import math
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path
from numpy.core._exceptions import _ArrayMemoryError
import pandas as pd
from ccramic.utils.region import RectangleRegion, FreeFormRegion
import os
from tifffile import imwrite
import numpy as np
import plotly.graph_objs as go

def generate_area_statistics_dataframe(graph_layout, upload, layers, data_selection, aliases_dict,
                                       zoom_keys = ['xaxis.range[1]', 'xaxis.range[0]',
                                                  'yaxis.range[1]', 'yaxis.range[0]'],
                                       modified_rect_keys=['shapes[1].x0', 'shapes[1].x1',
                                                           'shapes[1].y0', 'shapes[1].y1']):
    """
    Generate a Pandas Dataframe of channel information for selected region(s)
    Regions can be drawn on the `dcc.Graph` using zoom, rectangles, or closed freeform shapes with an svgpath
    """
    # option 1: if shapes are drawn on the canvas
    if 'shapes' in graph_layout and len(graph_layout['shapes']) > 0:
        # these are for each sample
        mean_panel = []
        max_panel = []
        min_panel = []
        aliases = []
        region = []
        region_index = 1
        shapes_keep = [shape for shape in graph_layout['shapes'] if shape['type'] not in ['line']]
        for shape in shapes_keep:
            try:
                # option 1: if the shape is drawn with a rectangle
                if shape['type'] == 'rect':
                    for layer in layers:
                        region_shape = RectangleRegion(upload[data_selection][layer], shape, reg_type="rect")
                        mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                        max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                        min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)
                        region.append(region_index)
                    # option 2: if a closed form shape is drawn
                elif shape['type'] == 'path' and 'path' in shape:
                    for layer in layers:
                        region_shape = FreeFormRegion(upload[data_selection][layer], shape)
                        mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                        max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                        min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                        aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)
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
                      'Region': region}
        return pd.DataFrame(layer_dict).to_dict(orient='records')

    # option 2: if the zoom is used
    elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
            all([elem in graph_layout for elem in zoom_keys]):

        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            aliases = []

            for layer in layers:
                region = RectangleRegion(upload[data_selection][layer], graph_layout, reg_type="zoom")
                mean_panel.append(round(float(region.compute_pixel_mean()), 2))
                max_panel.append(round(float(region.compute_pixel_max()), 2))
                min_panel.append(round(float(region.compute_pixel_min()), 2))
                aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

            return pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')

    # option 3: if a shape has already been created and is modified
    elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
            all([elem in graph_layout for elem in modified_rect_keys]):
        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            aliases = []

            for layer in layers:
                region_shape = RectangleRegion(upload[data_selection][layer], graph_layout,
                                               reg_type="rect", redrawn=True)
                mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)
            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

            return pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, _ArrayMemoryError, TypeError):
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')

    # option 4: if an svg path has already been created and it is modified
    elif ('shapes' not in graph_layout or len(graph_layout['shapes']) <= 0) and \
            all(['shapes' in elem and 'path' in elem for elem in graph_layout.keys()]):
        try:
            mean_panel = []
            max_panel = []
            min_panel = []
            aliases = []
            for layer in layers:
                for shape_path in graph_layout.values():
                    region_shape = FreeFormRegion(upload[data_selection][layer], shape_path)
                    mean_panel.append(round(float(region_shape.compute_pixel_mean()), 2))
                    max_panel.append(round(float(region_shape.compute_pixel_max()), 2))
                    min_panel.append(round(float(region_shape.compute_pixel_min()), 2))
                aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)
            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

            return pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError, AttributeError):
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')
    else:
        return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                             'Min': []}).to_dict(orient='records')


def output_current_canvas_as_tiff(canvas_image, dest_dir="/tmp/", output_file="canvas.tiff"):
    """
    Output the current canvas image as a photometric tiff
    """
    if canvas_image is not None:
        dest_file = str(os.path.join(dest_dir, output_file))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        imwrite(dest_file, canvas_image.astype(np.uint8), photometric='rgb')
        return dest_file
    else:
        return None

def output_current_canvas_as_html(cur_graph, canvas_style, dest_dir=None):
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
    fig.write_html(str(os.path.join(dest_dir, 'canvas.html')), default_width=canvas_style['width'],
                   default_height=canvas_style['height'])
    return str(os.path.join(dest_dir, 'canvas.html'))
