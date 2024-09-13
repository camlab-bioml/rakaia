from typing import Union
from functools import partial
import dash
import dash_bio
import pandas as pd
from dash import Patch
from pandas.errors import UndefinedVariableError
from pandas.core.dtypes.common import is_string_dtype
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np
from rakaia.parsers.object import drop_columns_from_measurements_csv
from rakaia.utils.object import subset_measurements_frame_from_umap_coordinates
from rakaia.utils.pixel import glasbey_palette


class PandasFrameSummaryModes:
    """
    Define the static pandas operations for summarizing a channel expression frame

    :return: None
    """
    @staticmethod
    def pd_mean(frame: pd.DataFrame):
        """
        :return: The data frame channel mean along the column axis
        """
        return frame.mean(axis=0)

    @staticmethod
    def pd_max(frame: pd.DataFrame):
        """
        :return: The data frame channel max along the column axis
        """
        return frame.max(axis=0)

    @staticmethod
    def pd_min(frame: pd.DataFrame):
        """
        :return: The data frame channel min along the column axis
        """
        return frame.min(axis=0)

    @staticmethod
    def pd_median(frame: pd.DataFrame):
        """
        :return: The data frame channel median along the column axis
        """
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

def clustergram_axes_order(clustergram: Union[dict, go.Figure]):
    """
    Get the order of channels and overlay sub types for a `dash_bio.Clustergram` for the figure data slots.
    Returns a tuple of orders for both the x and y-axis
    """
    x_order = None
    y_order = None
    clustergram = clustergram.to_dict() if not isinstance(clustergram, dict) else clustergram
    for key, val in clustergram['layout'].items():
        if isinstance(val, dict) and 'ticktext' in val and 'anchor' in val:
            x_order = val['ticktext'] if 'x' in val['anchor'] else x_order
            y_order = val['ticktext'] if 'y' in val['anchor'] else y_order
    return x_order, y_order

def grouped_heatmap(quantification: Union[dict, pd.DataFrame], umap_overlay: str, normalize: bool=True,
                    transpose: bool=False):
    """
    Generate a grouped channel heatmap of objects grouped by a umap overlay. Creates a mean expression
    summary of overlay subtypes x channels
    Normalization done by each channel, with the option to transpose for visualization purposes
    """
    quantification = pd.DataFrame(quantification)
    if not umap_overlay or umap_overlay not in quantification.columns: return None
    quantification[umap_overlay] = quantification[umap_overlay].apply(str)
    grouped = pd.DataFrame(quantification.groupby([umap_overlay]).mean())
    if normalize:
        grouped = pd.DataFrame((np.array(grouped) / np.array(grouped).max(axis=0)), columns=grouped.columns,
                           index=grouped.index)
    grouped = grouped.transpose() if transpose else grouped
    x_lab = list(grouped.columns)
    y_lab = [str(i) for i in pd.Series(grouped.index).to_list()]
    fig = go.Figure(dash_bio.Clustergram(
        data=grouped,
        column_labels=x_lab,
        row_labels=y_lab,
        height=500,
        width=750,
        color_map=[
                [0.0, "rgb(68, 1, 84)"],
                [0.5, "rgb(33, 145, 140)"],
                [1.0, "rgb(253, 231, 37)"],
            ]
        # standardize='column' if not invert else 'row'
    ))
    # fig.update_layout()
    x_order, y_order = clustergram_axes_order(fig)
    grouped = grouped.loc[:, y_order]
    reorder = grouped.reindex(x_order)

    # need to add check to ensure that the last data slot is always of a heatmap type?
    fig['data'][-1]['z'] = np.array(reorder)
    fig['layout']['title']['text'] = f"Channel Expression by {umap_overlay} ({len(quantification)} objects)"
    # zmax = 1 if np.max(grouped) <= 1 else np.max(grouped)
    # fig = go.Figure(px.imshow(np.array(grouped), x=grouped.columns, y=group_labs,
    #                           labels=dict(x="Channel", y=umap_overlay, color="Expression Mean"),
    #                           title=f"Channel Expression by {umap_overlay} ({len(quantification)} objects)",
    #                           zmax=zmax, binary_compression_level=1))
    fig.update_layout(margin={"pad": 0, "l": 2})
    fig.update_traces(hovertemplate="x: %{x} <br>y: %{y} <br>Expression: %{z} <br> <extra></extra>")
    return fig

def custom_channel_list_heatmap(measurements: pd.DataFrame, cols_include: Union[list, None]=None):
    """
    Specify a custom list of channels for the heatmap.
    """
    if cols_include is not None and len(cols_include) > 0 and \
            all(elem in measurements.columns for elem in cols_include):
        measurements = measurements[cols_include]
    return measurements

def generate_channel_heatmap(measurements, cols_include=None, drop_cols=False, subset_val=50000,
                             umap_overlay: Union[str, None]=None, normalize: bool=True, transpose: bool=False):
    """
    Generate a heatmap of the current quantification frame (total or subset)
    """
    measurements = pd.DataFrame(measurements)
    if umap_overlay and umap_overlay in measurements.columns:
        return grouped_heatmap(measurements, umap_overlay, normalize, transpose)
    if drop_cols:
        measurements = drop_columns_from_measurements_csv(measurements)
    measurements = custom_channel_list_heatmap(measurements, cols_include)
    # add a string to the title if subsampling is used
    total_objects = len(measurements)
    # default subset value is 50,000, for some reason the chart doesn't render properly after this many elements
    if subset_val is not None and isinstance(subset_val, int) and subset_val < len(measurements):
        measurements = measurements.sample(n=subset_val).reset_index(drop=True)
    array_measure = np.array(measurements)
    zmax = 1.0 if float(np.max(array_measure)) <= 1.0 else float(np.max(array_measure))
    fig = go.Figure(px.imshow(array_measure, x=measurements.columns, y=measurements.index,
                              labels=dict(x="Channel", y="Objects", color="Expression Mean"),
                              title=f"Channel expression ({len(measurements)}/{total_objects} shown)",
                              zmax=zmax, binary_compression_level=1, color_continuous_scale='viridis'))
    return fig

def generate_umap_plot(embeddings: Union[pd.DataFrame, dict, None], channel_overlay: Union[str, None]=None,
                       quantification_dict: Union[pd.DataFrame, dict, None]=None,
                       cur_umap_fig: Union[go.Figure, dict, None]=None,
                       categorical_size_limit: Union[int, float]=50) -> Union[go.Figure, dict]:
    """
    Generate a data frame of UMAP coordinates for a dataset of segmented objects based on channel expression
    Overlay color groupings can be passed as either numerical or categorical. Categorical variables are either
    strings as identified by pandas, or column with fewer subtypes than a threshold (recommended 50-100).
    """
    if embeddings is not None and len(embeddings) > 0:
        quant_frame = pd.DataFrame(quantification_dict)
        umap_frame = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
        palette = None
        try:
            if channel_overlay:
                if is_string_dtype(quant_frame[channel_overlay]) or \
                        len(quant_frame[channel_overlay].value_counts()) <= categorical_size_limit:
                    quant_frame[channel_overlay] = quant_frame[channel_overlay].apply(str)
                    palette = glasbey_palette(len(quant_frame[channel_overlay].value_counts()))
                umap_frame[channel_overlay] = quant_frame[channel_overlay]
            fig = px.scatter(umap_frame, x="UMAP1", y="UMAP2", color=channel_overlay,
                             color_discrete_sequence=palette)
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

def umap_eligible_patch(cur_umap_fig: Union[go.Figure, dict, None], quantification_dict: Union[pd.DataFrame, dict, None],
                        channel_overlay: Union[str, None], categorical_threshold: Union[int, float]=50):
    """
    Check if the current UMAP is available for a dash-style Patch for channel overlay
    Must already have a channel overlay applied so that the only updates to the figure
    are the color hovers over the channel intensities
    IMPORTANT: this only works for numeric to numeric overlay. Numeric to categorical or categorical to numeric
    requires a complete recreation of the figure because of the figure layout properties
    Overlays that have more than a threshold of unique values are treated as categorical
    """
    if cur_umap_fig:
        # numeric overlay does not have a legend group in the data slot
        # if switching to a categorical variable, do not patch (recreate)
        try:
            return 'data' in cur_umap_fig and 'layout' in cur_umap_fig and len(cur_umap_fig['data']) > 0 and \
                    not cur_umap_fig['data'][0]['legendgroup'] and \
                    cur_umap_fig['data'][0]['hovertemplate'] != 'UMAP1=%{x}<br>UMAP2=%{y}<extra></extra>' and \
            str(pd.DataFrame(quantification_dict)[channel_overlay].dtype) not in ["object"] and \
            len(pd.DataFrame(quantification_dict)[channel_overlay].value_counts()) >= categorical_threshold
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

def generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, mode_value,
                                               umap_layout, embeddings, zoom_keys, triggered_id, cols_drop=None,
                                                category_column=None, category_subset=None):
    """
    Generate a bar plot of summarized channel expression using interactive umap subsetting
    """
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # IMP: perform category subsetting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        if cols_drop is not None:
            frame = drop_columns_from_measurements_csv(frame)
        if triggered_id in ["umap-plot", "umap-projection-options", "quantification-bar-mode"] and \
                umap_layout is not None and \
                all(key in umap_layout for key in zoom_keys):
            subset_frame, overlay = subset_measurements_frame_from_umap_coordinates(frame,
                            pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2']),
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

def column_min_max_measurements(measurements: Union[dict, pd.DataFrame], normalize: bool=True):
    """
    Normalize a measurements data frame of summarized channel expression per object. Uses the column min-max
    to normalize the expression values between 0 and 1 per channel
    """
    if normalize:
        measurements = pd.DataFrame(measurements)
        measurements = ((measurements - measurements.min()) / (measurements.max() - measurements.min()))
    return measurements

def filter_overlay_from_heatmap_data(quant_frame: pd.DataFrame, overlay: Union[str, None]=None,
                                     categorical_size_limit: Union[int, float]=50):
    """
    Check a categorical UMAP overlay after subsetting to ensure that there are enough unique sub types to create
    a grouped heatmap. If not, remove the column
    """
    overlay_use = overlay if overlay and overlay in quant_frame.columns and \
    (1 < len(quant_frame[overlay].value_counts()) <= categorical_size_limit) else None
    if overlay in quant_frame.columns and not overlay_use:
        return quant_frame.drop([overlay], axis=1), overlay_use
    return quant_frame, overlay_use

def generate_heatmap_from_interactive_subsetting(quantification_dict: Union[dict, pd.DataFrame],
                                                umap_layout: dict, embeddings: Union[dict, pd.DataFrame],
                                                zoom_keys: list, triggered_id: str, cols_drop: bool=True,
                                                category_column: Union[str, None]=None,
                                                category_subset: Union[list, None]=None,
                                                cols_include: Union[list, None]=None,
                                                normalize=True, subset_val=None, umap_overlay: Union[str, None]=None,
                                                categorical_size_limit: Union[int, float]=30,
                                                transpose: bool=False):
    """
    Generate a heatmap of the quantification frame, trimmed to only the channel columns, based on an interactive
    subset from the UMAP graph
    """
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # use a umap overlay to group the heatmap only if it's categorical
        overlay_use = {umap_overlay: frame[umap_overlay]} if umap_overlay and \
                (1 < len(frame[umap_overlay].value_counts()) <= categorical_size_limit) else None
        # IMP: perform category sub-setting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        frame = drop_columns_from_measurements_csv(frame, cols_drop)
        # these are the columns that should be used to select the heatmap
        out_cols = frame.columns
        frame = custom_channel_list_heatmap(frame, cols_include)
        # need to normalize before the subset occurs so that it is relative to the entire frame, not just the subset
        frame = column_min_max_measurements(frame, normalize)
        frame, overlay_use = subset_measurements_frame_from_umap_coordinates(frame, pd.DataFrame(embeddings,
                            columns=['UMAP1', 'UMAP2']), umap_layout, umap_overlay=overlay_use)
        # need to check the value counts again after subsetting based on the restyler
        frame, overlay_use = filter_overlay_from_heatmap_data(frame, overlay_use, categorical_size_limit)
        # IMP: do not reset the subset index here as the indices are needed for the query subset!!!!
        # subset_frame = subset_frame.reset_index(drop=True)
        try:
            fig = generate_channel_heatmap(frame.reset_index(drop=True), cols_include=None,
                subset_val=subset_val, umap_overlay=overlay_use, normalize=normalize, transpose=transpose)
        except ValueError:
            fig = go.Figure()
        fig['layout']['uirevision'] = True
        # else:
        #     fig = dash.no_update
        return fig, frame, out_cols
    raise PreventUpdate

def reset_custom_gate_slider(trigger_id: str=None):
    """
    Reset the custom gate slider to False (not enabled) when certain triggers happen
    By default, the gating selection will come from the quantified parameters, and secondary is the custom list
    Do not reset the slider if the quantification hash has been updated on an annotation
    """
    return False if (trigger_id and trigger_id not in
            ["quantification-dict", "quantification_dict"]) else dash.no_update
