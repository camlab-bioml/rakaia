import os
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
import pytest
from rakaia.utils.dge import (
    dge_anndata,
    DGEOverlayMismatchError,
    anndata_requires_dge_subset)

def test_dge_anndata(get_current_dir):
    adata = os.path.join(get_current_dir, 'visium_thalamus.h5ad')
    assert not anndata_requires_dge_subset(adata)
    dge_results = dge_anndata(adata, 'array_col')
    assert int(dge_results.shape[0]) == 50
    assert int(dge_results.shape[1]) > 1
    assert not dge_anndata(None, 'leiden')
    assert not dge_anndata(adata, None)
    assert not dge_anndata(adata, 'not_a_column')

    cols_external = pd.Series(['fake'] * ad.read_h5ad(adata).shape[0])
    dge_results = dge_anndata(adata, cols_external)
    assert dge_results.shape == (50, 1)

    with pytest.raises(ValueError):
        dge_anndata(ad.read_h5ad(adata)[:2, :], 'array_col')

    with pytest.raises(DGEOverlayMismatchError):
        # when the overlay doesn't match the shape i.e. annotations that exclude objects
        assert not dge_anndata(adata, pd.Series(['fake'] * 100))

def test_dge_large():

    rng = np.random.default_rng(42)
    X = rng.random((50000, 10000), dtype=np.float32)

    adata = ad.AnnData(X=X)

    adata.obs["cluster"] = rng.choice(["group_1", "group_2"], size=50000)

    adata.obs_names = [f"cell_{i}" for i in range(50000)]
    adata.var_names = [f"gene_{i}" for i in range(10000)]

    assert anndata_requires_dge_subset(adata)
    dge_table = dge_anndata(adata, 'cluster')
    assert dge_table.shape == (50, 2)
