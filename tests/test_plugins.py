import os
import anndata
import pandas as pd
import numpy as np
import pytest
from rakaia.plugins import (
    PluginNotFoundError,
    run_quantification_model)
from rakaia.plugins.models import (
    QuantificationRandomForest,
    leiden_clustering)


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
                                                **{"n_estimators": 100, "max_depth": 8}).quantification_with_labels())
    assert not pd.Series(with_predictions["area_predict_100"]).equals(pd.Series(with_predictions["area_predict_25"]))

def test_run_quant_models(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    leiden_as_model = run_quantification_model(measurements_csv, None, resolution=0.5)
    assert "out" in pd.DataFrame(leiden_as_model).columns
    rf_model = run_quantification_model(leiden_as_model, "area", "area_predict", "random forest", n_estimators=25)
    assert "area_predict" in pd.DataFrame(rf_model).columns

    with pytest.raises(PluginNotFoundError):
        run_quantification_model(measurements_csv, None, mode="not found")
