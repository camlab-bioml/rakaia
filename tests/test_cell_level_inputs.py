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
