from typing import Union
import pandas as pd
import plotly.express as px
from pandas.core.dtypes.common import is_string_dtype
# from pandas.core.dtypes.common import is_numeric_dtype
import plotly.graph_objs as go

from rakaia.utils.pixel import glasbey_palette

def metadata_association_plot(metadata_frame: Union[pd.DataFrame, dict, list],
                              x_axis_variable: str,
                              y_axis_variable: str,
                              grouping: str=None,
                              retain_uirevision: bool=False,
                              categorical_size_limit: Union[int, float]=50):
    """
    Produces either a `plotly.express` scatter plot or violin plot of summarized metadata variables
    Plots may either have a grouping or no grouping
    Categories are treated as categorical if they are parsed as strings by pandas, or
    if they have fewer unique values than a categorical limit (recommended 50-100).
    """
    metadata_frame = pd.DataFrame(metadata_frame)
    palette = None
    if grouping and len(metadata_frame[grouping].value_counts()) <= categorical_size_limit:
        metadata_frame[grouping] = metadata_frame[grouping].apply(str)
        palette = glasbey_palette(len(metadata_frame[grouping].value_counts()))
    # if the x-axis variable is categorical, make it violin. otherwise, make scatter
    # categorical if it's a string OR there are fewer than 100 unique values
    if is_string_dtype(metadata_frame[x_axis_variable]) or \
        (len(metadata_frame[x_axis_variable].value_counts()) <= categorical_size_limit):
        metadata_frame[x_axis_variable] = metadata_frame[x_axis_variable].apply(str)
        grouping = grouping if (grouping and len(metadata_frame[grouping].value_counts()) <=
                                categorical_size_limit) else None
        palette = palette if grouping else None
        fig = go.Figure(px.violin(metadata_frame, y=y_axis_variable,
                                   x=x_axis_variable, color=grouping, box=True,
                                  color_discrete_sequence=palette))
    else:
        fig = go.Figure(px.scatter(metadata_frame, y=y_axis_variable,
                                   x=x_axis_variable, color=grouping,
                                   color_discrete_sequence=palette))
    fig['layout']['uirevision'] = retain_uirevision
    return fig
