from typing import Union
from functools import partial
import plotly.graph_objs as go
from rakaia.plugins.models import leiden_clustering

class PluginNotFoundError(Exception):
    """
    Raises an exception when a particular plugin model mode is not found
    """

class PluginDescriptors:
    """
    Defines the set of plugin descriptors that are available in the application dropdown

    :return: None
    """
    # the descriptors should be the string representation of the model in the application dropdown
    descriptors = ["leiden"]

class PluginModes:
    """
    Define the set of modes that are selectable by the application plugin dropdown


    :return: None
    """
    # each model here should be named as the string representation of PluginDescriptors.descriptors,
    # but with spaces replaced with underscores
    # Each should return a modified quantification sheet
    @staticmethod
    def leiden(quantification_results: Union[dict, go.Figure], input_category: str,
                  output_category: str):
        return leiden_clustering(quantification_results, output_category)


class PluginModelModes:
    """
    Define the partial functions for the selectable plugin models

    :return: None
    """
    # each attribute should be directly named after a specific function in PluginModes, and should be a partial function
    leiden = partial(PluginModes.leiden)

def run_quantification_model(quantification_results: Union[dict, go.Figure], input_category: Union[str, None]=None,
                             output_category: str="out", mode: str="leiden"):
    """
    Run a quantification model on a set of objects with quantified channel intensities. Requires a specific model
    mode provided by the user
    """
    # convert the mode to a spaceless string representation to match the attribute
    mode = mode.replace(" ", "_") if mode else ""
    if mode not in PluginDescriptors.descriptors:
        raise PluginNotFoundError(f"The plugin mode provided: {mode} is not a supported plugin."
                                  f"Current plugins include: {PluginDescriptors.descriptors}")
    with_model = getattr(PluginModelModes, mode)(quantification_results, input_category, output_category)
    return with_model
