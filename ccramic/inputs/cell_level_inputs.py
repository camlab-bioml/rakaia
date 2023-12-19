import dash
import pandas as pd

from ccramic.parsers.cell_level_parsers import drop_columns_from_measurements_csv
from ccramic.utils.cell_level_utils import subset_measurements_frame_from_umap_coordinates
from pandas.errors import UndefinedVariableError
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

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

    if drop_cols:
        dropped = drop_columns_from_measurements_csv(measurement_frame)
    else:
        dropped = pd.DataFrame(measurement_frame)
    if mode == "mean":
        dropped = dropped.mean(axis=0)
    elif mode == "max":
        dropped = dropped.max(axis=0)
    elif mode == "min":
        dropped = dropped.min(axis=0)

    summary_frame = pd.DataFrame(dropped, columns=[mode]).rename_axis("Channel").reset_index()
    if len(summary_frame) > 0 and summary_frame is not None:
        return px.bar(summary_frame, x="Channel", y=str(mode), color="Channel",
                      title=f"Segmented Marker Expression ({len(measurement_frame)} cells)")
    else:
        return None

def generate_channel_heatmap(measurements, cols_include=None, drop_cols=True):
    """
    Generate a heatmap of the current quantification frame (total or subset)
    """
    measurements = pd.DataFrame(measurements)
    if drop_cols:
        measurements = drop_columns_from_measurements_csv(measurements)
    if cols_include is not None and len(cols_include) > 0 and \
            all([elem in measurements.columns for elem in cols_include]):
        measurements = measurements[cols_include]
    return px.imshow(measurements, x=measurements.columns, y=measurements.index,
                    labels=dict(x="Channel", y="Cells", color="Expression Mean"),
                    title=f"Channel expression per cell ({len(measurements)} cells)")

def generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
    if embeddings is not None and len(embeddings) > 0:
        quant_frame = pd.DataFrame(quantification_dict)
        df = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
        try:
            if channel_overlay is not None:
                df[channel_overlay] = quant_frame[channel_overlay]
            fig = px.scatter(df, x="UMAP1", y="UMAP2", color=channel_overlay)
        except KeyError:
            fig = px.scatter(df, x="UMAP1", y="UMAP2")
        if cur_umap_fig is None:
            fig['layout']['uirevision'] = True
        else:
            fig['layout'] = cur_umap_fig['layout']
            fig['layout']['uirevision'] = True
        return fig
    else:
        return dash.no_update

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
        # TODO: for now, o not allow the bar plot to reflect a canvas subset (assign values only from the UMAP)
        # if canvas_layout is not None and \
        #         all([key in canvas_layout for key in zoom_keys]) and triggered_id == "annotation_canvas":
        #     try:
        #         subset_zoom = {"x_min": min(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
        #                    "x_max": max(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
        #                    "y_min": min(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]']),
        #                    "y_max": max(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]'])}
        #     except UndefinedVariableError:
        #         subset_zoom = None
        #     fig = go.Figure(get_cell_channel_expression_plot(frame, subset_dict=subset_zoom, mode=mode_value))
        #     frame_return = frame
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
    else:
        raise PreventUpdate


def generate_heatmap_from_interactive_subsetting(quantification_dict, umap_layout, embeddings, zoom_keys,
                                                triggered_id, cols_drop=True,
                                                category_column=None, category_subset=None, cols_include=None,
                                                normalize=True):
    """
    Generate a heatmap of the quantification frame, trimmed to only the channel columns, based on an interactive
    subset from the UMAP graph
    """
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # IMP: perform category subsetting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        if cols_drop:
            frame = drop_columns_from_measurements_csv(frame)
        # TODO: important: need to normalize before  the subset occurs so that it is relative to the entire
        # frame, not just the subset
        # TODO: add min max normalization to have ranges between 0 and 1
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
            fig = generate_channel_heatmap(subset_frame.reset_index(drop=True), cols_include=cols_include)
            fig['layout']['uirevision'] = True
        else:
            fig = dash.no_update
        return fig, subset_frame
    else:
        raise PreventUpdate
