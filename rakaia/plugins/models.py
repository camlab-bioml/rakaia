from typing import Union
import anndata
import pandas as pd
import scanpy as sc

from rakaia.parsers.object import (
    quant_dataframe_to_anndata,
    parse_quantification_sheet_from_h5ad)

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
