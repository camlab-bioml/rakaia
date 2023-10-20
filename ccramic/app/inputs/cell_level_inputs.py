import pandas as pd

from ..inputs.cell_level_inputs import *
from ..utils.cell_level_utils import *
from ..parsers.cell_level_parsers import *
from pandas.errors import UndefinedVariableError

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

def generate_umap_plot(embeddings, channel_overlay, quantification_dict, cur_umap_fig):
    if embeddings is not None:
        quant_frame = pd.DataFrame(quantification_dict)
        df = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
        if channel_overlay is not None:
            df[channel_overlay] = quant_frame[channel_overlay]
        try:
            fig = px.scatter(df, x="UMAP1", y="UMAP2", color=channel_overlay)
        except KeyError:
            fig = px.scatter(df, x="UMAP1", y="UMAP2")
        if cur_umap_fig is None:
            fig['layout']['uirevision'] = True
        else:
            fig['layout'] = cur_umap_fig['layout']
        return fig
    else:
        raise PreventUpdate

def generate_expression_bar_plot_from_interactive_subsetting(quantification_dict, canvas_layout, mode_value,
                                               umap_layout, embeddings, zoom_keys, triggered_id, cols_drop=None,
                                                category_column=None, category_subset=None):
    if quantification_dict is not None and len(quantification_dict) > 0:
        frame = pd.DataFrame(quantification_dict)
        # IMP: perform category subsetting before removing columns
        if None not in (category_column, category_subset):
            frame = frame[frame[category_column].isin(category_subset)]
        if cols_drop is not None:
            frame = drop_columns_from_measurements_csv(frame, cols_to_drop=cols_drop)
        if all([key in canvas_layout for key in zoom_keys]) and triggered_id == "annotation_canvas":
            try:
                subset_zoom = {"x_min": min(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                           "x_max": max(canvas_layout['xaxis.range[0]'], canvas_layout['xaxis.range[1]']),
                           "y_min": min(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]']),
                           "y_max": max(canvas_layout['yaxis.range[0]'], canvas_layout['yaxis.range[1]'])}
            except UndefinedVariableError:
                subset_zoom = None
            fig = go.Figure(get_cell_channel_expression_plot(frame, subset_dict=subset_zoom, mode=mode_value))
        elif triggered_id in ["umap-plot", "umap-projection-options"] and \
                all([key in umap_layout for key in zoom_keys]):
            subset_frame = subset_measurements_frame_from_umap_coordinates(frame,
                                                                           pd.DataFrame(embeddings,
                                                                                        columns=['UMAP1', 'UMAP2']),
                                                                           umap_layout)
            fig = go.Figure(get_cell_channel_expression_plot(subset_frame,
                                                             subset_dict=None, mode=mode_value))
        else:
            subset_zoom = None
            fig = go.Figure(get_cell_channel_expression_plot(frame,
                                                             subset_dict=subset_zoom, mode=mode_value))
        fig['layout']['uirevision'] = True
        return fig
    else:
        raise PreventUpdate
