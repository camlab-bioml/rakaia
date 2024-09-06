from typing import Union
import anndata
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rakaia.parsers.object import (
    quant_dataframe_to_anndata,
    parse_quantification_sheet_from_h5ad)

class QuantificationRandomForest:
    """
    Run a random forest classifier on a set of quantified objects with channel intensities. Requires a column
    specifying an existing annotation that will be used to predict the class for the remaining objects in the
    dataset that have not been annotated

    :param quantification: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
    :param in_col: Existing annotation column in the object that specifies the classes used to fit the model
    :param out_col: Column to store the output labels of the classifier prediction
    :return: None
    """
    def __init__(self, quantification: Union[dict, pd.DataFrame], in_col: str, out_col: str="rf"):
        self.quantification = pd.DataFrame(quantification)
        self.in_col = in_col
        self.out_col = out_col
        self.clf = RandomForestClassifier(random_state=100)
        # since this is the default in the app, ignore any samples with just this annotation
        self.null_cats = ["Unassigned"]
        self.sample_identifier = "description" if "description" in list(self.quantification.columns) else "sample"
        self.training_samples = self.set_training_samples()
        self.training_subset = self.set_training_subset()
        self.labels = None

    def set_training_samples(self):
        """
        Set the identifiers for the training samples. Training samples are identified as those with existing
        annotations that are not `Unassigned`

        :return: List of sample identifiers to create the training subset
        """
        sam_train = []
        for sam in self.quantification[self.sample_identifier].unique():
            subset_measure = self.quantification[self.quantification[self.sample_identifier] == sam]
            annots_sub = [annot for annot in subset_measure[self.in_col].unique() if annot not in self.null_cats]
            if annots_sub:
                sam_train.append(sam)
        return sam_train

    def set_training_subset(self):
        """

        :return: Training subset from the quantification using the list of training samples
        """
        return quant_dataframe_to_anndata(
            self.quantification[self.quantification[self.sample_identifier].isin(self.training_samples)])

    def run_model(self):
        """
        Run the model (fit the classifier to the training subset, and generate the prediction labels on the entire
        dataset).

        :return: None
        """
        self.clf.fit(np.array(self.training_subset.X), self.training_subset.obs[self.in_col])
        self.labels = self.clf.predict(np.array(quant_dataframe_to_anndata(self.quantification).X))

    def quantification_with_labels(self, return_as_dict: bool=True):
        """

        :param return_as_dict: Whether to return the quantification table as a record-oriented dictionary or not.
        :return: Quantification results with the prediction labels added under the output column.
        """
        if not self.labels:
            self.run_model()
        self.quantification[self.out_col] = self.labels
        return self.quantification.to_dict(orient="records") if return_as_dict else self.quantification

def leiden_clustering(quantification: Union[pd.DataFrame, anndata.AnnData, dict],
                    out_col: str="leiden", n_neighbors: Union[int, float]=30, return_as_dict: bool=True):
    """
    Apply leiden clustering to a set of quantified channel intensities. Accepts either an `anndata.AnnData` object
    or a `pd.DataFrame`. Converts the input into an `anndata.AnnData` object before processing
    Returns either an `anndata.AnnData` object or a dictionary representation of a `pd.DataFrame`
    """
    if not isinstance(quantification, anndata.AnnData):
        quantification = quant_dataframe_to_anndata(quantification)
    sc.pp.neighbors(quantification, n_neighbors=n_neighbors)
    sc.tl.leiden(quantification, key_added=out_col)
    return parse_quantification_sheet_from_h5ad(quantification).to_dict(
        orient="records") if return_as_dict else quantification
