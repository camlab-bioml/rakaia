from typing import Union
import dash
import pandas as pd
from dash import Patch
from pandas.errors import UndefinedVariableError
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np
from functools import partial
from rakaia.parsers.object import drop_columns_from_measurements_csv
from rakaia.utils.object import subset_measurements_frame_from_umap_coordinates

class PandasFrameSummaryModes:
    @staticmethod
    def pd_mean(frame: pd.DataFrame):
        return frame.mean(axis=0)

    @staticmethod
    def pd_max(frame: pd.DataFrame):
        return frame.max(axis=0)

    @staticmethod
    def pd_min(frame: pd.DataFrame):
        return frame.min(axis=0)

    @staticmethod
    def pd_median(frame: pd.DataFrame):
        return frame.median(axis=0)

class BarChartPartialModes:
    """
    Defines a series of pandas summary statistics for the bar chart
    Pandas operators are applied using partials with a provided measurement frame
    """
    mean = partial(PandasFrameSummaryModes.pd_mean)
    max = partial(PandasFrameSummaryModes.pd_max)
    min = partial(PandasFrameSummaryModes.pd_min)
    median = partial(PandasFrameSummaryModes.pd_median)

def get_cell_channel_expression_plot(measurement_frame, mode="mean",
                                     subset_dict=None, drop_cols=True):
    """
    Generate a bar plot of the expression of channels by cell for a specific metric (mean, max, min, etc.)
    Ensure that the non-numeric columns are dropped prior to plotting
    """
    if subset_dict is not None and len(subset_dict) == 4:
        try:
            measurement_frame = measurement_frame.query(f'x_max >= {subset_dict["x_min"]} & '
                                                        f'x_max <= {subset_dict["x_max"]} & '
                                                        f'y_max >= {subset_dict["y_min"]} & '
                                                        f'y_max <= {subset_dict["y_max"]}')
        except (KeyError, UndefinedVariableError):
            pass
    dropped = pd.DataFrame(measurement_frame)
    if drop_cols:
        dropped = drop_columns_from_measurements_csv(measurement_frame)
    # call the partial using the mode and provided measurement frame
    dropped = getattr(BarChartPartialModes, mode)(dropped)
    summary_frame = pd.DataFrame(dropped, columns=[mode]).rename_axis("Channel").reset_index()
    if len(summary_frame) > 0 and summary_frame is not None:
        return px.bar(summary_frame, x="Channel", y=str(mode), color="Channel",
                      title=f"Segmented Marker Expression ({len(measurement_frame)} cells)")
    return None

def generate_channel_heatmap(measurements, cols_include=None, drop_cols=True, subset_val=50000):
    """
    Generate a heatmap of the current quantification frame (total or subset)
    """
    measurements = pd.DataFrame(measurements)
    if drop_cols:
        measurements = drop_columns_from_measurements_csv(measurements)
    if cols_include is not None and len(cols_include) > 0 and \
            all([elem in measurements.columns for elem in cols_include]):
        measurements = measurements[cols_include]
    # add a string to the title if subsampling is used
    total_objects = len(measurements)
    if subset_val is not None and isinstance(subset_val, int) and subset_val < len(measurements):
        measurements = measurements.sample(n=subset_val).reset_index(drop=True)
    # TODO: figure out why the colour bars won't render after a certain number of dataframe elements
    array_measure = np.array(measurements)
    zmax = 1 if np.max(array_measure) <= 1 else np.max(array_measure)
    fig = go.Figure(px.imshow(array_measure, x=measurements.columns, y=measurements.index,
                              labels=dict(x="Channel", y="Objects", color="Expression Mean"),
                              title=f"Channel expression per object ({len(measurements)}/{total_objects} shown)",
                              zmax=zmax, binary_compression_level=1))
    return fig

def generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
    if embeddings is not None and len(embeddings) > 0:
        quant_frame = pd.DataFrame(quantification_dict)
        umap_frame = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
        try:
            if channel_overlay is not None:
                umap_frame[channel_overlay] = quant_frame[channel_overlay]
            fig = px.scatter(umap_frame, x="UMAP1", y="UMAP2", color=channel_overlay)
        except KeyError:
            fig = px.scatter(umap_frame, x="UMAP1", y="UMAP2")
            fig['data'][0]['showlegend'] = True
        if cur_umap_fig is None:
            fig['layout']['uirevision'] = True
        else:
            fig['layout'] = cur_umap_fig['layout']
            fig['layout']['uirevision'] = True
        return fig
    return dash.no_update

def umap_eligible_patch(cur_umap_fig: Union[go.Figure, dict], quantification_dict: Union[pd.DataFrame, dict],
                        channel_overlay: str):
    """
    Check if the current UMAP is available for a dash-style Patch for channel overlay
    Must already have a channel overlay applied so that the only updates to the figure
    are the color hovers over the channel intensities
    IMPORTANT: this only works for numeric to numeric overlay. Numeric to categorical or categorical to numeric
    requires a complete recreation of the figure because of the figure layout properties
    """
    if cur_umap_fig:
        # numeric overlay does not have a legend group in the data slot
        # if switching to a categorical variable, do not patch (recreate)
        try:
            return 'data' in cur_umap_fig and 'layout' in cur_umap_fig and len(cur_umap_fig['data']) > 0 and \
                    not cur_umap_fig['data'][0]['legendgroup'] and \
                    cur_umap_fig['data'][0]['hovertemplate'] != 'UMAP1=%{x}<br>UMAP2=%{y}<extra></extra>' and \
            str(pd.DataFrame(quantification_dict)[channel_overlay].dtype) not in ["object"]
        except KeyError:
            return False
    return False

def patch_umap_figure(quantification_dict: Union[pd.DataFrame, dict], channel_overlay: str):
    """
    Patch an existing UMAP channel plot with a new overlay of channel expression
    """
    patched_figure = Patch()
    # patched_figure['layout']['coloraxis']['colorbar']['title']['text'] = channel_overlay
    patched_figure['data'][0]['marker']['color'] = pd.DataFrame(quantification_dict)[channel_overlay]
    patched_figure['data'][0]['hovertemplate'] = 'UMAP1=%{x}<br>UMAP2=%{y}<br>' + \
                                                 channel_overlay + '=%{marker.color}<extra></extra>'
    return patched_figure

def generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, zoom_keys, triggered_id, cols_drop=None,
                                                category_column=None, category_subset=None):
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # IMP: perform category subsetting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        if cols_drop is not None:
            frame = drop_columns_from_measurements_csv(frame)
        if triggered_id in ["umap-plot", "umap-projection-options", "quantification-bar-mode"] and \
                umap_layout is not None and \
                all([key in umap_layout for key in zoom_keys]):
            subset_frame = subset_measurements_frame_from_umap_coordinates(frame,
                                                                           pd.DataFrame(embeddings,
                                                                                        columns=['UMAP1', 'UMAP2']),
                                                                           umap_layout)
            fig = go.Figure(get_cell_channel_expression_plot(subset_frame,
                                                             subset_dict=None, mode=mode_value))
            frame_return = subset_frame
        else:
            subset_zoom = None
            fig = go.Figure(get_cell_channel_expression_plot(frame,
                                                             subset_dict=subset_zoom, mode=mode_value))
            frame_return = frame
        fig['layout']['uirevision'] = True
        return fig, frame_return
    raise PreventUpdate


def generate_heatmap_from_interactive_subsetting(quantification_dict, umap_layout, embeddings, zoom_keys,
                                                triggered_id, cols_drop=True,
                                                category_column=None, category_subset=None, cols_include=None,
                                                normalize=True, subset_val=None):
    """
    Generate a heatmap of the quantification frame, trimmed to only the channel columns, based on an interactive
    subset from the UMAP graph
    """
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # IMP: perform category sub-setting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        if cols_drop:
            frame = drop_columns_from_measurements_csv(frame)
        # need to normalize before  the subset occurs so that it is relative to the entire frame, not just the subset
        if normalize:
            frame = ((frame - frame.min()) / (frame.max() - frame.min()))
        if umap_layout is not None and \
                all([key in umap_layout for key in zoom_keys]):
            subset_frame = subset_measurements_frame_from_umap_coordinates(frame,
                        pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2']), umap_layout)
        else:
            subset_frame = frame
        # IMP: do not reset the subset index here as the indices are needed for the query subset!!!!
        # subset_frame = subset_frame.reset_index(drop=True)
        # only recreate the graph if new data are passed from the UMAP, not on a recolouring of the UMAP
        if triggered_id not in ["umap-projection-options"] or category_column is None:
            # IMP: reset the index for the heatmap to avoid uneven box sizes
            try:
                fig = generate_channel_heatmap(subset_frame.reset_index(drop=True), cols_include=cols_include,
                                           subset_val=subset_val)
            except ValueError:
                fig = go.Figure()
            fig['layout']['uirevision'] = True
        else:
            fig = dash.no_update
        return fig, subset_frame
    raise PreventUpdate

def reset_custom_gate_slider(trigger_id: str=None):
    """
    Reset the custom gate slider to False (not enabled) when certain triggers happen
    By default, the gating selection will come from the quantified parameters, and secondary is the custom list
    Do not reset the slider if the quantification hash has been updated on an annotation
    """
    return False if (trigger_id and trigger_id not in
            ["quantification-dict", "quantification_dict"]) else dash.no_update
