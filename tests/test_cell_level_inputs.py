import pytest

from ccramic.app.inputs.cell_level_inputs import *
import os
import plotly.graph_objs as go
import pandas as pd
import plotly
from ccramic.app.parsers.cell_level_parsers import *
from dash.exceptions import PreventUpdate

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
    validated_measurements = parse_and_validate_measurements_csv(measurements_dict)
    umap_plot = generate_umap_plot(umap_dict, None, validated_measurements, None)
    assert isinstance(umap_plot, plotly.graph_objs._figure.Figure)
    assert umap_plot['layout']['uirevision']
    with pytest.raises(PreventUpdate):
        generate_umap_plot(None, None, None, None)

def test_expression_plot_from_interactive_triggers(get_current_dir):
    measurements_dict = {"uploads": [os.path.join(get_current_dir, "cell_measurements.csv")]}
    validated_measurements, cols = parse_and_validate_measurements_csv(measurements_dict)
    umap_dict = {"UMAP1": [1, 2, 3, 4, 5, 6], "UMAP2": [6, 7, 8, 9, 10, 11]}
    zoom_keys = ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']
    interactive_umap = generate_expression_bar_plot_from_interactive_subsetting(validated_measurements, {}, "mean",
                                                                                {}, umap_dict, zoom_keys, "umap_plot")
    assert '(244 cells)' in interactive_umap['layout']['title']['text']
    subset_layout = {'xaxis.range[0]': 400, 'xaxis.range[1]': 900, 'yaxis.range[0]': 65, 'yaxis.range[1]': 5}
    interactive_umap = generate_expression_bar_plot_from_interactive_subsetting(validated_measurements, {},
                                                                                "mean", subset_layout,
                                                                                umap_dict, zoom_keys, "umap_plot")
    assert interactive_umap['layout']['uirevision']
