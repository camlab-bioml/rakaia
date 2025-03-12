import os
import collections
import anndata
import pandas as pd
import numpy as np
import pytest
import anndata as ad
from rakaia.plugins import (
    PluginNotFoundError,
    run_quantification_model)
from rakaia.plugins.models import (
    QuantificationRandomForest,
    leiden_clustering,
    ObjectMixingRF,
    AdaBoostTreeClassifier,
    subset_anndata_by_var_names)

def test_subset_anndata_channel_names(get_current_dir):
    expr = ad.read_h5ad(os.path.join(get_current_dir, 'quantification_anndata.h5ad'))
    keep_full = subset_anndata_by_var_names(expr)
    assert keep_full.shape == expr.shape
    subset_some = subset_anndata_by_var_names(expr, ['ERK1/2', 'Lamin B1',
       'Mitochondria'])
    assert subset_some.shape == (1445, 3)
    assert subset_anndata_by_var_names(expr,
            ['not_real_col', 'Mitochondria']).shape == expr.shape

def test_leiden_clustering(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    assert "leiden" not in measurements_csv.columns
    with_leiden = leiden_clustering(measurements_csv, n_neighbors=10, resolution=0.5)
    assert "leiden" in pd.DataFrame(with_leiden).columns
    with_leiden_adata = leiden_clustering(measurements_csv, n_neighbors=10, return_as_dict=False)
    assert isinstance(with_leiden_adata, anndata.AnnData)
    assert "leiden" in with_leiden_adata.obs.columns

def test_quant_random_forest(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    # predict the area for test_2
    measurements_csv['area_use'] = np.where(measurements_csv['sample'] == 'test_1', measurements_csv['area'], 'Unassigned')
    with_predictions = QuantificationRandomForest(measurements_csv, "area_use", "area_predict_25",
                                                  n_estimators=25).quantification_with_labels()
    assert 'area_predict_25' in pd.DataFrame(with_predictions).columns
    assert 'Unassigned' not in pd.DataFrame(with_predictions)['area_predict_25']
    with_predictions = pd.DataFrame(QuantificationRandomForest(with_predictions, "area_use", "area_predict_100",
                                                **{"max_depth": 8, "n_estimators": 100}).quantification_with_labels())
    assert not pd.Series(with_predictions["area_predict_100"]).equals(pd.Series(with_predictions["area_predict_25"]))

def test_quant_boosting(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    # predict the area for test_2
    measurements_csv['area_use'] = np.where(measurements_csv['sample'] == 'test_1', measurements_csv['area'],
                                            'Unassigned')
    with_predictions = AdaBoostTreeClassifier(measurements_csv, "area_use", "area_predict_25",
                                                ).quantification_with_labels()
    assert 'area_predict_25' in pd.DataFrame(with_predictions).columns
    assert 'Unassigned' not in pd.DataFrame(with_predictions)['area_predict_25']
    with_predictions = pd.DataFrame(QuantificationRandomForest(with_predictions, "area_use", "area_predict_100",
                                                               **{"max_depth": 8,
                                                                  "n_estimators": 100}).quantification_with_labels())
    assert not pd.Series(with_predictions["area_predict_100"]).equals(pd.Series(with_predictions["area_predict_25"]))


def test_object_mixing(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    for prop in [0.2, 0.25, 0.3, 0.35]:
        rf_model = ObjectMixingRF(measurements_csv, None, "mixing", training_set_prop=prop,
                                                  n_estimators=100)
        # assert that the training length and dataset length are the same, as the training takes half of the data
        # randomly then adds mixed cells
        assert collections.Counter(list(rf_model.training_labels)) == {0.0: int(len(measurements_csv) * prop),
                                                                       1.0: int(len(measurements_csv) * prop)}
        assert len(rf_model.training_labels) == (int(len(measurements_csv) * prop) * 2)
        with_predictions = rf_model.quantification_with_labels()
        assert 'mixing' in pd.DataFrame(with_predictions).columns
        assert sum(pd.Series(pd.DataFrame(with_predictions)['mixing'])) > 0

def test_run_quant_models(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    leiden_as_model = run_quantification_model(measurements_csv, None, resolution=0.5)
    assert "out" in pd.DataFrame(leiden_as_model).columns

    rf_model = run_quantification_model(leiden_as_model, "area", "area_predict", "random forest", n_estimators=25)
    assert "area_predict" in pd.DataFrame(rf_model).columns

    mixing = run_quantification_model(rf_model, None, "mixing", "object mixing", n_estimators=100)
    assert "mixing" in pd.DataFrame(mixing).columns

    adaboost = run_quantification_model(mixing, 'area', "boosted", "boosted trees", n_estimators=1,
                                        max_depth=4)
    assert "boosted" in pd.DataFrame(adaboost).columns

    assert not pd.DataFrame(adaboost)['boosted'].equals(pd.DataFrame(adaboost)['area_predict'])

    with pytest.raises(PluginNotFoundError):
        run_quantification_model(measurements_csv, None, mode="not found")
