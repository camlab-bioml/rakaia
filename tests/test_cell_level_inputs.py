from ccramic.app.inputs.cell_level_inputs import *
import os
import plotly.graph_objs as go
import pandas as pd

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
