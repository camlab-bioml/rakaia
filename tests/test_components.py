import numpy as np
from ccramic.components.canvas import CanvasImage
import plotly.graph_objs as go
import plotly.express as px

def test_basic_canvas_image():

    canvas_layers = {"roi_1": {"channel_1": np.full((100, 100), 10),
                               "channel_2": np.full((100, 100), 20), "channel_3": np.full((100, 100), 30)}}
    currently_selected = ["channel_1", "channel_2", "channel_3"]
    data_selection = "roi_1"
    mask_config = {"roi_1": {"array": np.full((100, 100), 1), "boundary": np.zeros((100, 100))}}
    mask_selection = "roi_1"
    mask_blending_level = 40
    overlay_grid = []
    mask_toggle = True
    add_mask_boundary = True
    invert_annot = True
    cur_graph = go.Figure()
    pixel_ratio = None
    legend_text = ''
    toggle_scalebar = True
    legend_size = 12
    toggle_legend = True
    add_cell_id_hover = []
    show_each_channel_intensity = []
    raw_data_dict = canvas_layers
    aliases = {"channel_1": "first", "channel_2": "second", "channel_3": "third"}

    canvas = CanvasImage(canvas_layers, data_selection, currently_selected,
                mask_config, mask_selection, mask_blending_level,
                overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                show_each_channel_intensity, raw_data_dict, aliases)
    assert isinstance(canvas, CanvasImage)
    canvas_fig = canvas.generate_canvas()
    assert isinstance(canvas_fig, go.Figure)

    # overlay_grid = [' overlay grid']
    add_cell_id_hover = [' show mask ID on hover']
    show_each_channel_intensity = [" show channel intensities on hover"]

    canvas_2 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases)
    assert isinstance(canvas_2, CanvasImage)
    canvas_fig = canvas_2.generate_canvas()
    assert isinstance(canvas_fig, go.Figure)


    cur_graph = px.imshow(canvas.get_image())
    # overlay_grid = [' overlay grid']
    add_cell_id_hover = [' show mask ID on hover']
    show_each_channel_intensity = [" show channel intensities on hover"]
    canvas_3 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases)
    canvas_fig_3 = canvas_3.generate_canvas()
    assert isinstance(canvas_fig_3, go.Figure)


    cur_graph = {'data': {'customdata': np.full((100, 100), 10)}, 'layout': {'uirevision': True}}
    canvas_4 = CanvasImage(canvas_layers, data_selection, currently_selected,
                           mask_config, mask_selection, mask_blending_level,
                           overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                           legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                           show_each_channel_intensity, raw_data_dict, aliases)

    canvas_fig_4 = canvas_4.generate_canvas()
    assert isinstance(canvas_fig_4, go.Figure)

    cur_graph = {'data': {'customdata': None}, 'layout': {'uirevision': True}}
    canvas_5 = CanvasImage(canvas_layers, data_selection, currently_selected,
                           mask_config, mask_selection, mask_blending_level,
                           overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                           legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                           show_each_channel_intensity, raw_data_dict, aliases)

    canvas_fig_5 = canvas_5.generate_canvas()
    assert isinstance(canvas_fig_5, go.Figure)

    add_cell_id_hover = []
    canvas_6 = CanvasImage(canvas_layers, data_selection, currently_selected,
                           mask_config, mask_selection, mask_blending_level,
                           overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                           legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                           show_each_channel_intensity, raw_data_dict, aliases)
    canvas_fig_6 = canvas_6.generate_canvas()
    assert isinstance(canvas_fig_6, go.Figure)

    cur_graph = {'data': {'customdata': None}, 'layout': {'uirevision': True, 'shapes': [{'fake_key': 'fake_val'}]}}
    canvas_7 = CanvasImage(canvas_layers, data_selection, currently_selected,
                           mask_config, mask_selection, mask_blending_level,
                           overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                           legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                           show_each_channel_intensity, raw_data_dict, aliases)
    canvas_fig_7 = canvas_7.generate_canvas()
    assert isinstance(canvas_fig_7, go.Figure)
