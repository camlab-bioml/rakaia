import numpy as np
from ccramic.components.canvas import CanvasImage, CanvasLayout
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import os
from PIL import Image

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
    global_apply_filter = [' Apply/refresh filter']
    global_filter_type = "gaussian"
    global_filter_val = 5
    global_filter_sigma = 1
    apply_cluster_on_mask = False
    cluster_assignments_dict = None
    cluster_frame = None
    cluster_type = 'mask'
    custom_scale_val = None

    canvas = CanvasImage(canvas_layers, data_selection, currently_selected,
                mask_config, mask_selection, mask_blending_level,
                overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask, cluster_assignments_dict,
                         cluster_frame, cluster_type, custom_scale_val)
    assert isinstance(canvas, CanvasImage)
    canvas_fig = canvas.generate_canvas()
    assert isinstance(canvas_fig, dict)

    global_filter_type = "median"
    global_filter_val = 33
    custom_scale_val = 48

    canvas = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                         global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                         cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    assert isinstance(canvas, CanvasImage)
    canvas_fig = canvas.generate_canvas()
    assert isinstance(canvas_fig, dict)




    # overlay_grid = [' overlay grid']
    add_cell_id_hover = [' Show mask ID on hover']
    show_each_channel_intensity = [" Show channel intensities on hover"]

    canvas_2 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    assert isinstance(canvas_2, CanvasImage)
    canvas_fig = canvas_2.generate_canvas()
    assert isinstance(canvas_fig, dict)


    cur_graph = px.imshow(canvas.get_image())
    # overlay_grid = [' overlay grid']
    add_cell_id_hover = [' Show mask ID on hover']
    show_each_channel_intensity = [" Show channel intensities on hover"]
    canvas_3 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    canvas_fig_3 = canvas_3.generate_canvas()
    assert isinstance(canvas_fig_3, dict)


    cur_graph = {'data': {'customdata': np.full((100, 100), 10)}, 'layout': {'uirevision': True}}
    canvas_4 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)

    canvas_fig_4 = canvas_4.generate_canvas()
    assert isinstance(canvas_fig_4, dict)

    cur_graph = {'data': {'customdata': None}, 'layout': {'uirevision': True}}
    canvas_5 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)

    canvas_fig_5 = canvas_5.generate_canvas()
    assert isinstance(canvas_fig_5, dict)

    add_cell_id_hover = []
    canvas_6 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    canvas_fig_6 = canvas_6.generate_canvas()
    assert isinstance(canvas_fig_6, dict)

    cur_graph = {'data': {'customdata': None}, 'layout': {'uirevision': True, 'shapes': [{'fake_key': 'fake_val'}]}}
    canvas_7 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    canvas_fig_7 = canvas_7.generate_canvas()
    assert isinstance(canvas_fig_7, dict)

    canvas_layers = {"roi_1": {"channel_1": np.full((100, 100, 3), 10),
                               "channel_2": np.full((100, 100, 3), 20), "channel_3": np.full((100, 100, 3), 30)}}
    cluster_frame = {"roi_1": pd.DataFrame({'cell_id': list(range(1, 10, 1)),
                                 'cluster': ['Cluster_1'] * 9})}
    cluster_assignments_dict = {"roi_1": {"Cluster_1": '#FFFFFF'}}
    apply_cluster_on_mask = True
    mask_config = {"roi_1": {"array": np.full((100, 100, 3), 1), "boundary": np.zeros((100, 100, 3)),
                             "raw": np.full((100, 100), 1).astype(np.float32)}}
    overlay_grid = [' Overlay grid']
    canvas_8 = CanvasImage(canvas_layers, data_selection, currently_selected,
                         mask_config, mask_selection, mask_blending_level,
                         overlay_grid, mask_toggle, add_mask_boundary, invert_annot, cur_graph, pixel_ratio,
                         legend_text, toggle_scalebar, legend_size, toggle_legend, add_cell_id_hover,
                         show_each_channel_intensity, raw_data_dict, aliases, global_apply_filter, global_filter_type,
                 global_filter_val, global_filter_sigma, apply_cluster_on_mask,
                           cluster_assignments_dict, cluster_frame, cluster_type, custom_scale_val)
    canvas_fig_8 = canvas_8.generate_canvas()
    assert isinstance(canvas_fig_8, dict)


def test_canvas_layout_editor(get_current_dir):
    image = np.full((600, 600, 3), 255).astype(np.uint8)
    fig = go.Figure(px.imshow(image))
    assert len(fig['layout']['shapes']) == 0
    # fig = {'data': fig['data'], 'layout': fig['layout']}
    fig = CanvasLayout(fig)
    point_annotations = pd.read_csv(os.path.join(get_current_dir, "point_annotations.csv"))
    fig = fig.add_point_annotations_as_circles(point_annotations, image, 4)
    assert len(fig['layout']['shapes']) > 0

    fig = go.Figure(px.imshow(image))
    assert len(fig['layout']['annotations']) == 0

    fig = go.Figure(CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 1, image.shape, 12))
    assert len(fig['layout']['annotations']) > 0

    # update the layout to mimic a zoom to change the scalebar value
    fig.update_layout(xaxis=dict(range=[50, 60]), yaxis=dict(range=[50, 60]))
    fig = CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 1, image.shape, 12)
    assert len(fig['layout']['annotations']) > 0

    fig = go.Figure(px.imshow(image))
    fig = CanvasLayout(fig).toggle_scalebar(False, 0.05, True, 1, image.shape, 12)
    assert len(fig['layout']['annotations']) == 0


    fig = go.Figure(px.imshow(image))
    fig = CanvasLayout(fig).toggle_legend(True, "legend_text", 0.99, 14)
    assert 'legend_text' in fig['layout']['annotations'][0]['text']

    fig = {'data': fig['data'], 'layout': {'annotations': [{
        'bgcolor': 'black',
        'font': {'size': 15},
        'showarrow': False,
        'text': 'legend_text: color',
        'x': 0.010000000000000009,
        'xref': 'paper',
        'y': 0.05,
        'yref': 'paper'
    },
        {
            'bgcolor': 'black',
            'font': {'size': 15},
            'showarrow': False,
            'text': 'legend',
            'x': 0.010000000000000009,
            'xref': 'paper',
            'y': 0.06,
            'yref': 'paper'
        }
    ]}}
    fig = CanvasLayout(fig).change_annotation_size(9)
    assert 'legend_text' in fig['layout']['annotations'][0]['text']

    fig = go.Figure(px.imshow(image))
    fig = CanvasLayout(fig).toggle_legend(False, "legend_text", 0.99, 14)
    assert len(fig['layout']['annotations']) == 0

    fig = go.Figure(px.imshow(image))
    fig.update_layout(xaxis=dict(range=[50, 60]), yaxis=dict(range=[50, 60]))
    fig = CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 0, image.shape, 12)
    fig = CanvasLayout(fig).update_scalebar_zoom_value({"test": "test"}, 1)
    assert 'color: white">2μm</span><br>' in fig['layout']['annotations'][0]['text']

    fig = go.Figure(px.imshow(image))
    canvas_layout = {'xaxis.range[1]': 50, 'xaxis.range[0]': 60}
    fig = CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 1, image.shape, 12)
    fig = CanvasLayout(fig).update_scalebar_zoom_value(canvas_layout, 1)
    assert 'color: white">2μm</span><br>' in fig['layout']['annotations'][0]['text']

    shapes = [{'line': {'color': 'white', 'width': 2}, 'type': 'line', 'x0': 0.875, 'x1': 0.95, 'xref': 'paper', 'y0': 0.05, 'y1': 0.05, 'yref': 'paper'}, {'editable': True, 'line': {'color': 'white'}, 'type': 'circle', 'x0': 763, 'x1': 779, 'xref': 'x', 'y0': 284, 'y1': 300, 'yref': 'y', 'label': {'texttemplate': ''}}, {'editable': True, 'line': {'color': 'white'}, 'type': 'circle', 'x0': 796, 'x1': 812, 'xref': 'x', 'y0': 293, 'y1': 309, 'yref': 'y'}]
    fig = go.Figure(px.imshow(image))
    fig_dict = {'data': fig['data'], 'layout': {
    'margin': {'t': 60},
    'template': '...',
    'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0]},
    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0]},
    'shapes' : []}}

    fig_dict['layout']['shapes'] = shapes
    fig = CanvasLayout(fig_dict).clear_improper_shapes()
    assert len(fig['layout']['shapes']) == len(shapes)
    for shape in fig['layout']['shapes']:
        if len(shape) == 1:
            assert 'label' not in shape

    image = np.full((1079, 1095, 3), 255).astype(np.uint8)
    clusters = pd.read_csv(os.path.join(get_current_dir, "cluster_assignments.csv"))
    colors = {"roi_1": {"Type_1": "red", "Type_2": "blue", "Type_3": "yellow"}}
    fig = go.Figure(px.imshow(image))
    mask = np.array(Image.open(os.path.join(get_current_dir, "mask.tiff")))
    fig = CanvasLayout(fig).add_cluster_annotations_as_circles(mask, clusters, colors, "roi_1")
    assert len(fig['layout']['shapes']) > 1000

    fig = CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 1, image.shape, 12)
    fig = CanvasLayout(fig).remove_cluster_annotation_shapes()
    assert len(fig['layout']['shapes']) == 1

    fig = go.Figure(px.imshow(image))
    # canvas_layout = {'xaxis.range[1]': 50, 'xaxis.range[0]': 60}
    fig = CanvasLayout(fig).toggle_scalebar(True, 0.05, True, 1, image.shape, 12)
    fig = CanvasLayout(fig).use_custom_scalebar_value(23, 1)
    assert '<span style="color: white">23μm</span><br>' in fig['layout']['annotations'][0]['text']

    window_dict = {'y_low': 100, 'y_high': 150, 'x_low': 100, 'x_high': 150}
    fig = go.Figure(px.imshow(image))
    fig, window_layout = CanvasLayout(fig).update_coordinate_window(window_dict, 250, 250)
    assert window_layout == {'xaxis.range[0]': 225.0, 'xaxis.range[1]': 275.0,
                             'yaxis.range[0]': 275.0, 'yaxis.range[1]': 225.0}
    assert fig['layout']['xaxis']['range'] == [225.0, 275.0]
    assert fig['layout']['yaxis']['range'] == [275.0, 225.0]
