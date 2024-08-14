from typing import Union
import pandas as pd
import plotly.express as px
from pandas.core.dtypes.common import is_string_dtype
# from pandas.core.dtypes.common import is_numeric_dtype
import plotly.graph_objs as go

def metadata_association_plot(metadata_frame: Union[pd.DataFrame, dict],
                              x_axis_variable: str,
                              y_axis_variable: str,
                              grouping: str=None,
                              retain_uirevision: bool=False,
                              grouping_size_limit: Union[int, float]=100):
    """
    Produces either a scatter plot or violin plot of summarized metadata variables
    Plots may either have a grouping or no grouping.
    """
    metadata_frame = pd.DataFrame(metadata_frame)
    grouping = grouping if (grouping and len(metadata_frame[grouping].value_counts()) <= grouping_size_limit) else None
    # if the x-axis variable is categorical, make it violin. otherwise, make scatter
    if is_string_dtype(metadata_frame[x_axis_variable]):
        fig = go.Figure(px.violin(metadata_frame, y=y_axis_variable,
                                   x=x_axis_variable, color=grouping, box=True))
    else:
        fig = go.Figure(px.scatter(metadata_frame, y=y_axis_variable,
                                   x=x_axis_variable, color=grouping))
    fig['layout']['uirevision'] = retain_uirevision
    return fig
