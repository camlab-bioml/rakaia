from ccramic.utils.alert import (
    AlertMessage, file_import_message,
    DataImportTour)
import os

def test_basic_alerts():
    alert_config = AlertMessage().warnings
    assert len(alert_config) == 12

def test_tour_steps():
    tour_steps = DataImportTour().steps
    assert isinstance(tour_steps, list)
    assert len(tour_steps) == 4
    assert 'upload-image' in tour_steps[0]['selector']

def test_file_import_warning(get_current_dir):
    files = [os.path.join(get_current_dir, "data.h5"), os.path.join(get_current_dir, "query.mcd")]
    message, unique = file_import_message(files)
    assert message == 'Read in the following files:\n'\
                      '/home/matt/github/ccramic/tests/data.h5\n'\
                      '/home/matt/github/ccramic/tests/query.mcd\n'
    assert len(unique) == 2