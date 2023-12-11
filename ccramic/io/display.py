import math
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path
from numpy.core._exceptions import _ArrayMemoryError
import pandas as pd
from ccramic.utils.region import RectangleRegion, FreeFormRegion

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
            assert all([elem >= 0 for elem in graph_layout.keys() if isinstance(elem, float)])
            x_range_low = math.ceil(int(graph_layout['shapes[1].x0']))
            x_range_high = math.ceil(int(graph_layout['shapes[1].x1']))
            y_range_low = math.ceil(int(graph_layout['shapes[1].y0']))
            y_range_high = math.ceil(int(graph_layout['shapes[1].y1']))
            assert x_range_high >= x_range_low
            assert y_range_high >= y_range_low

            mean_panel = []
            max_panel = []
            min_panel = []
            aliases = []
            for layer in layers:
                mean_exp, max_xep, min_exp = get_area_statistics_from_rect(upload[data_selection][layer],
                                                                           x_range_low,
                                                                           x_range_high,
                                                                           y_range_low, y_range_high)
                mean_panel.append(round(float(mean_exp), 2))
                max_panel.append(round(float(max_xep), 2))
                min_panel.append(round(float(min_exp), 2))
                aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

            return pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, _ArrayMemoryError):
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
                    mean_exp, max_xep, min_exp = get_area_statistics_from_closed_path(
                        upload[data_selection][layer], shape_path)
                    mean_panel.append(round(float(mean_exp), 2))
                    max_panel.append(round(float(max_xep), 2))
                    min_panel.append(round(float(min_exp), 2))
                aliases.append(aliases_dict[layer] if layer in aliases_dict.keys() else layer)

            layer_dict = {'Channel': aliases, 'Mean': mean_panel, 'Max': max_panel, 'Min': min_panel}

            return pd.DataFrame(layer_dict).to_dict(orient='records')

        except (AssertionError, ValueError, ZeroDivisionError, TypeError, _ArrayMemoryError):
            return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                                 'Min': []}).to_dict(orient='records')
    else:
        return pd.DataFrame({'Channel': [], 'Mean': [], 'Max': [],
                             'Min': []}).to_dict(orient='records')
