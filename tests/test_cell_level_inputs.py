import dash
import pytest

import os
import plotly.graph_objs as go
import pandas as pd
import plotly
from dash.exceptions import PreventUpdate
from ccramic.inputs.cell_level_inputs import (
    get_cell_channel_expression_plot,
    generate_umap_plot,
    generate_expression_bar_plot_from_interactive_subsetting,
    generate_channel_heatmap,
    generate_heatmap_from_interactive_subsetting)
from ccramic.parsers.cell_level_parsers import parse_and_validate_measurements_csv


def test_bar_graph_from_measurements_csv(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    cell_bar = get_cell_channel_expression_plot(measurements_csv)
    assert isinstance(cell_bar, go.Figure)
    assert cell_bar['layout']['xaxis']['title']['text'] == "Channel"
    assert cell_bar['layout']['yaxis']['title']['text'] == "mean"

    cell_bar_max = get_cell_channel_expression_plot(measurements_csv, mode="max")
    assert cell_bar_max['layout']['xaxis']['title']['text'] == "Channel"
    assert cell_bar_max['layout']['yaxis']['title']['text'] == "max"
    assert cell_bar_max['layout']['title']['text'] == 'Segmented Marker Expression (244 cells)'

    cell_bar_min = get_cell_channel_expression_plot(measurements_csv, mode="min")
    assert cell_bar_min['layout']['xaxis']['title']['text'] == "Channel"
    assert cell_bar_min['layout']['yaxis']['title']['text'] == "min"
    assert cell_bar_min['layout']['title']['text'] == 'Segmented Marker Expression (244 cells)'


def test_bar_graph_from_measurements_csv_with_subsetting(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    subset_dict = {"x_max": 900, "x_min": 400, "y_max": 65, "y_min": 5}
    cell_bar = get_cell_channel_expression_plot(measurements_csv, subset_dict=subset_dict)
    assert '61 cells' in cell_bar['layout']['title']['text']

def test_umap_plot(get_current_dir):
    umap_dict = {"UMAP1": [1, 2, 3, 4, 5, 6], "UMAP2": [6, 7, 8, 9, 10, 11]}
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols, warning = parse_and_validate_measurements_csv(measurements_dict)
    assert isinstance(warning, dash._callback.NoUpdate)
    umap_plot = generate_umap_plot(umap_dict, None, validated_measurements, None)
    assert isinstance(umap_plot, plotly.graph_objs._figure.Figure)
    assert umap_plot['layout']['uirevision']
    with pytest.raises(PreventUpdate):
        generate_umap_plot(None, None, None, None)

def test_expression_plot_from_interactive_triggers(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols, warning = parse_and_validate_measurements_csv(measurements_dict)
    umap_dict = {"UMAP1": [1, 2, 3, 4, 5, 6], "UMAP2": [6, 7, 8, 9, 10, 11]}
    zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
    category_column = "sample"
    interactive_umap, frame = generate_expression_bar_plot_from_interactive_subsetting(validated_measurements, {}, "mean",
                                                                                {}, umap_dict, zoom_keys, None,
                                                                                category_column=category_column)
    assert '(244 cells)' in interactive_umap['layout']['title']['text']
    # subset_layout = {'xaxis.range[0]': 400, 'xaxis.range[1]': 900, 'yaxis.range[0]': 65, 'yaxis.range[1]': 5}
    # interactive_umap, frame = generate_expression_bar_plot_from_interactive_subsetting(validated_measurements, subset_layout,
    #                                                                             "mean", subset_layout,
    #                                                                     umap_dict, zoom_keys, "annotation_canvas")
    # assert interactive_umap['layout']['uirevision']
    # assert '(61 cells)' in interactive_umap['layout']['title']['text']
    umap_dict = {"UMAP1": list(range(900)), "UMAP2": list(range(900))}
    subset_layout = {'xaxis.range[0]': 400, 'xaxis.range[1]': 800, 'yaxis.range[0]': 65, 'yaxis.range[1]': 5}
    interactive_umap, frame = generate_expression_bar_plot_from_interactive_subsetting(validated_measurements, subset_layout,
                                                                                "mean", subset_layout,
                                                                                umap_dict, zoom_keys,
                                                                                "umap-plot")
    assert interactive_umap['layout']['uirevision']
    assert '(0 cells)' in interactive_umap['layout']['title']['text']


def test_quantification_heatmap(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    fig = generate_channel_heatmap(measurements_csv)
    # assert that the last element in the list of columns in the heatmap is a channel
    assert isinstance(fig, plotly.graph_objs._figure.Figure)
    assert list(fig['data'][0]['x'])[-1] == "209Bi_SMA"
    assert '(244 cells)' in fig['layout']['title']['text']

def test_heatmap_from_interactive_triggers(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols, warning = parse_and_validate_measurements_csv(measurements_dict)
    umap_dict = {"UMAP1": [1, 2, 3, 4, 5, 6], "UMAP2": [6, 7, 8, 9, 10, 11]}
    zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
    # category_column = "sample"
    interactive_heat, frame = generate_heatmap_from_interactive_subsetting(validated_measurements, {}, umap_dict,
                                                                zoom_keys, None, "umap-layout")
    assert '(244 cells)' in interactive_heat['layout']['title']['text']
    embeddings = pd.read_csv(os.path.join(get_current_dir, "umap_coordinates_for_measurements.csv"))
    subset_layout = {'xaxis.range[0]': 7.386287234198646, 'xaxis.range[1]': 9.393588084462001,
                     'yaxis.range[0]': 6.270861713114755, 'yaxis.range[1]': 9.579169008196722}
    interactive_heat, frame = generate_heatmap_from_interactive_subsetting(validated_measurements, subset_layout,
                                                                        embeddings, zoom_keys, "umap-layout")
    assert interactive_heat['layout']['uirevision']
    assert '(41 cells)' in interactive_heat['layout']['title']['text']


    subset_layout = {'xaxis.range[0]': 400, 'xaxis.range[1]': 800, 'yaxis.range[0]': 65, 'yaxis.range[1]': 5}
    interactive_heat, frame = generate_heatmap_from_interactive_subsetting(validated_measurements, subset_layout,
                                                                        embeddings, zoom_keys, "umap-projection-options",
                                                                        category_column="sample",
                                                                           category_subset=["test_1", "test_2"])
    assert isinstance(interactive_heat, dash._callback.NoUpdate)
