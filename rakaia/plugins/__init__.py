"""Module containing functions and classes for the plugin/object model API
"""
from typing import Union
from functools import partial
import pandas as pd
import plotly.graph_objs as go
from rakaia.plugins.models import (
    QuantificationRandomForest,
    leiden_clustering,
    ObjectMixingRF, AdaBoostTreeClassifier)

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
    descriptors = ["leiden", "random forest", "boosted trees", "object mixing"]

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
               output_category: str, var_name_subset: Union[str, None]=None, **kwargs):
        """
        Generate leiden clustering

        :param quantification_results: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
        :param input_category: Placeholder argument for similar plugins. NOt used for leiden clustering
        :param output_category: Column to store the outputs of the clustering procedure
        :param var_name_subset: List of channel names to subset, if specified
        :param kwargs: Keyword arguments to pass to `scanpy.tl.leiden`

        :return: Quantification results with the cluster labels added under the output column.
        """
        return leiden_clustering(quantification_results, var_name_subset,
                                 output_category, **kwargs)

    @staticmethod
    def random_forest(quantification_results: Union[dict, go.Figure], input_category: str,
                  output_category: str, var_name_subset: Union[str, None]=None, **kwargs):
        """
        Train and predict using a random forest classifier

        :param quantification_results: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
        :param input_category: Existing annotation column in the object that specifies the classes used to fit the model
        :param output_category: Column to store the output labels of the classifier prediction
        :param var_name_subset: List of channel names to subset, if specified
        :param kwargs: Keyword arguments to pass to `RandomForestClassifier`

        :return: Quantification results with the prediction labels added under the output column.
        """
        return QuantificationRandomForest(
            quantification_results, input_category, output_category, var_name_subset,
            **kwargs).quantification_with_labels(True)

    @staticmethod
    def object_mixing(quantification_results: Union[dict, go.Figure], input_category: str,
                      output_category: str, var_name_subset: Union[str, None]=None, **kwargs):
        """
        Train and predict object segmentation errors using object expression mixing.

        :param quantification_results: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
        :param input_category: Existing annotation column in the object that specifies the classes used to fit the model
        :param output_category: Column to store the output labels of the classifier prediction
        :param var_name_subset: List of channel names to subset, if specified
        :param kwargs: Keyword arguments to pass to `RandomForestClassifier`

        :return: Quantification results with the prediction labels added under the output column.
        """

        return ObjectMixingRF(
            quantification_results, input_category, output_category, var_name_subset,
            **kwargs).quantification_with_labels(True)

    @staticmethod
    def boosted_trees(quantification_results: Union[dict, go.Figure], input_category: str,
                      output_category: str, var_name_subset: Union[str, None]=None, **kwargs):
        """
        Train and predict using an Adaboost decision tree classifier

        :param quantification_results: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
        :param input_category: Existing annotation column in the object that specifies the classes used to fit the model
        :param output_category: Column to store the output labels of the classifier prediction
        :param var_name_subset: List of channel names to subset, if specified
        :param kwargs: Keyword arguments to pass to `AdaBoostClassifier`

        :return: Quantification results with the prediction labels added under the output column.
        """
        return AdaBoostTreeClassifier(
            quantification_results, input_category, output_category, var_name_subset,
            **kwargs).quantification_with_labels(True)


class PluginModelModes:
    """
    Define the partial functions for the selectable plugin models

    :return: None
    """
    # each attribute should be directly named after a specific function in PluginModes, and should be a partial function
    leiden = partial(PluginModes.leiden)
    random_forest = partial(PluginModes.random_forest)
    object_mixing = partial(PluginModes.object_mixing)
    boosted_trees = partial(PluginModes.boosted_trees)

def run_quantification_model(quantification_results: Union[dict, pd.DataFrame], input_category: Union[str, None]=None,
                             output_category: str="out", mode: str="leiden",
                             var_name_subset: Union[list, None]=None, **kwargs):
    """
    Run a quantification model on a set of objects with quantified channel intensities. Requires a specific model
    mode provided by the user
    """
    # convert the mode to a spaceless string representation to match the attribute
    if mode not in PluginDescriptors.descriptors:
        raise PluginNotFoundError(f"The plugin mode provided: {mode} is not a supported plugin."
                                  f"Current plugins include: {PluginDescriptors.descriptors}")
    # get the attribute compatible string for each mode from the dropdown menu
    mode = mode.replace(" ", "_") if mode else ""
    with_model = getattr(PluginModelModes, mode)(quantification_results, input_category,
                        output_category, var_name_subset, **kwargs)
    return with_model
