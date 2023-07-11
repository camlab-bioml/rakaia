
import pandas as pd
import plotly.express as px

def get_cell_channel_expression_plot(measurement_frame, mode="mean",
                                     dropped_columns=['cell_id', 'x', 'y', 'x_max', 'y_max', 'area', 'sample'],
                                     subset_dict=None):
    """
    generate a barplot of the expression of channels by cell for a specific metric (mean, max, min, etc.)
    """

    if subset_dict is not None and len(subset_dict) == 4:
        try:
            measurement_frame = measurement_frame.query(f'x_max >= {subset_dict["x_min"]} & '
                                                        f'x_max <= {subset_dict["x_max"]} & '
                                                        f'y_max >= {subset_dict["y_min"]} & '
                                                        f'y_max <= {subset_dict["y_max"]}')
        except KeyError:
            pass
    dropped = pd.DataFrame(measurement_frame).drop(dropped_columns, axis=1)
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
