from ccramic.app.parsers.cell_level_parsers import *
import numpy as np
import pandas as pd
import os

def test_validation_of_measurements_csv(get_current_dir):
    measurements_csv = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    assert measurements_csv.equals(validate_incoming_measurements_csv(measurements_csv))

    measurements_bad = measurements_csv.drop(['cell_id', 'x', 'y', 'x_max', 'y_max', 'area'], axis=1)
    assert validate_incoming_measurements_csv(measurements_bad) is None

    fake_image = np.empty((1490, 93, 3))
    assert validate_incoming_measurements_csv(measurements_csv, current_image=fake_image) is not None

    fake_image_bad_dims = np.empty((1490, 92, 3))
    assert validate_incoming_measurements_csv(measurements_csv, current_image=fake_image_bad_dims) is None
