import os
from rakaia.utils.dge import dge_anndata

def test_dge_anndata(get_current_dir):
    adata = os.path.join(get_current_dir, 'visium_thalamus.h5ad')
    dge_results = dge_anndata(adata, 'cluster')
    assert not (dge_results is None)
    assert dge_results.shape == (50, 1)
    dge_results = dge_anndata(adata, 'array_col')
    assert int(dge_results.shape[0]) == 50
    assert int(dge_results.shape[1]) > 1
    assert not dge_anndata(None, 'leiden')
    assert not dge_anndata(adata, None)
    assert not dge_anndata(adata, 'not_a_column')
