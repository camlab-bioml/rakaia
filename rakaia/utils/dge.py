"""
Module for classes and functions related to differential gene expression (DGE) for either
`Anndata` objects or numerical `numpy` arrays with a categorical grouping (i.e. `pandas.Series`)
"""
from typing import Union
import pandas as pd
import anndata as ad
import scanpy as sc
import uuid

class DGEOverlayMismatchError(Exception):
    """
    Raise when DGE cannot be performed with the given overlay because its shape does not match
    """

def dge_anndata(adata: Union[None, str, ad.AnnData],
                grouping: Union[None, str, list, pd.Series]=None,
                num_genes_show: int=50,
                gene_ranking_method: Union[str, None]="wilcoxon",
                min_group_size: int=2):
    """
    Perform differential gene expression (DGE) for a single `Anndata` object based on a categorical grouping
    present in the `adata.obs`
    """
    adata = ad.read_h5ad(adata) if isinstance(adata, str) else adata
    # IMP: this only works if the overlay has the same length as the expression i.e. not missing any objects
    if type(grouping) in (list, pd.Series) and not (adata is None):
        if not len(grouping) == adata.shape[0]:
            raise DGEOverlayMismatchError(f"DGE cannot be performed with the given grouping (dimensions do not match).")
        column_name = str(uuid.uuid4())
        adata.obs[column_name] = list(grouping)
        grouping = column_name
    if not (adata is None) and str(grouping) in adata.obs.columns:
        adata.obs[grouping] = adata.obs[str(grouping)].astype(str).astype("category")
        counts = adata.obs[str(grouping)].value_counts()
        valid_groups = counts[counts > min_group_size].index
        adata = adata[adata.obs[str(grouping)].isin(valid_groups)].copy()

        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=str(grouping), method=gene_ranking_method, use_raw=False)

        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        # gives only the genes without test statistics    # use specified gene number unless fewer exist
        return pd.DataFrame({group: result["names"][group][:min(num_genes_show,
                            int(len(adata.var_names)))] for group in groups})
    return None
