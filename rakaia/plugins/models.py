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
    :param kwargs: Keyword arguments to pass to `RandomForestClassifier`

    :return: None
    """
    def __init__(self, quantification: Union[dict, pd.DataFrame], in_col: Union[str, None]= None,
                 out_col: str="rf", **kwargs):
        self.quantification = pd.DataFrame(quantification)
        self.in_col = in_col if in_col else "sample"
        self.out_col = out_col
        self.clf = RandomForestClassifier(**kwargs)
        # since this is the default in the app, ignore any samples with just this annotation
        self.null_cats = ["Unassigned"]
        self.sample_identifier = "description" if "description" in list(self.quantification.columns) else "sample"
        self.training_samples = self.set_training_samples()
        self.training_subset, self.training_labels = self.set_training_subset()
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

        :return: Training subset data and labels from the quantification using the list of training samples
        """
        subset = quant_dataframe_to_anndata(
            self.quantification[self.quantification[self.sample_identifier].isin(self.training_samples)])
        labels = subset.obs[self.in_col]
        return subset, labels

    def run_model(self):
        """
        Run the model (fit the classifier to the training subset, and generate the prediction labels on the entire
        dataset).

        :return: None
        """
        training = np.array(self.training_subset.X if isinstance(self.training_subset,
                    anndata.AnnData) else self.training_subset)
        self.clf.fit(training, self.training_labels)
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

class ObjectMixingRF(QuantificationRandomForest):
    """
    Run a random forest classifier for classifying possibly mixed/mis-segmented objects on a set of quantified
    objects with channel intensities.

    :param quantification: `pd.DataFrame` or `anndata.AnnData` object containing tabular object intensity measurements
    :param in_col: Existing annotation column in the object that specifies the classes used to fit the model
    :param out_col: Column to store the output labels of the classifier prediction
    :param training_set_prop: Proportion of the input expression data to use for training. Default is 0.5 for 50%.
    :param kwargs: Keyword arguments to pass to `RandomForestClassifier`

    :return: None
    """
    def __init__(self, quantification: Union[dict, pd.DataFrame], in_col: Union[str, None]= None, out_col: str="rf",
                 training_set_prop: Union[int, float]=0.5, **kwargs):
        self.prop = training_set_prop
        super().__init__(quantification, in_col, out_col, **kwargs)

    def set_training_samples(self):
        """
        Set the identifiers for the training samples. For object mixing all ROI identifiers are used, then sub-sampled.

        :return: List of sample identifiers to create the training subset
        """
        # for cell mixing, just use all of the samples and then randomly subset
        return list(self.quantification[self.sample_identifier].unique())

    def set_training_subset(self, prop_mix: Union[int, float, None]=1.0):
        """
        Use the class training proportion and number of objects to mix to generate a training subset.

        :param prop_mix: If passed, sets the proportion number of cells to randomly mix relative to the negative training set.

        :return: Training subset data and labels from the quantification using the list of training samples
        """
        adata = quant_dataframe_to_anndata(
            self.quantification[self.quantification[self.sample_identifier].isin(self.training_samples)])

        # use the proportion to randomly subsample from the training data
        mask = np.random.choice([False, True], len(adata), p=[(1 - self.prop), self.prop])
        expr = adata.X[mask]

        num_sam, num_features = expr.shape

        n_mix = int(prop_mix * len(expr)) if prop_mix else num_sam
        new_x = np.zeros((n_mix, num_features))
        # create the training labels, first with the negative, then positive mixed
        y_train = np.concatenate([np.zeros(expr.shape[0]), np.ones(n_mix)])

        for n in range(n_mix):
            # pick two random indices
            idx1 = np.random.randint(0, num_sam)
            idx2 = np.random.randint(0, num_sam)

            # average expression of two merged cells
            new_x[n] = (expr[idx1] + expr[idx2]) / 2.

        x_train = np.concatenate([expr, new_x])

        return x_train, y_train

def leiden_clustering(quantification: Union[pd.DataFrame, anndata.AnnData, dict],
                    out_col: str="leiden", n_neighbors: Union[int, float]=30,
                    return_as_dict: bool=True, **kwargs):
    """
    Apply leiden clustering to a set of quantified channel intensities. Accepts either an `anndata.AnnData` object
    or a `pd.DataFrame`. Converts the input into an `anndata.AnnData` object before processing
    Returns either an `anndata.AnnData` object or a dictionary representation of a `pd.DataFrame`
    Accepts kwargs to pass to `scanpy.tl.leiden`.
    """
    if not isinstance(quantification, anndata.AnnData):
        quantification = quant_dataframe_to_anndata(quantification)
    sc.pp.neighbors(quantification, n_neighbors=n_neighbors)
    sc.tl.leiden(quantification, key_added=out_col, **kwargs)
    return parse_quantification_sheet_from_h5ad(quantification).to_dict(
        orient="records") if return_as_dict else quantification
