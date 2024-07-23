from rakaia.utils.alert import (
    AlertMessage, file_import_message,
    DataImportTour,
    ToolTips,
    add_warning_to_error_config)
import os

def test_basic_alerts():
    alert_config = AlertMessage().warnings
    assert len(alert_config) > 0
    assert all([isinstance(elem, str) for elem in alert_config])

def test_tour_steps():
    tour_steps = DataImportTour().steps
    assert isinstance(tour_steps, list)
    assert len(tour_steps) == 6
    assert 'upload-image' in tour_steps[0]['selector']

def test_file_import_warning(get_current_dir):
    files = [os.path.join(get_current_dir, "data.h5"), os.path.join(get_current_dir, "query.mcd")]
    message, unique = file_import_message(files)
    assert message == f'Read in the following files:\n'\
                        f'{os.path.join(get_current_dir, "data.h5")}\n'\
                        f'{os.path.join(get_current_dir, "query.mcd")}\n'\
                        '\n'\
                        ' Select a region (ROI) from the data collection dropdown to begin analysis.'
    assert len(unique) == 2


def test_tooltips():
    assert isinstance(ToolTips().tooltips, dict)
    assert "delimiter" in ToolTips().tooltips.keys()

def test_parse_add_error_config():
    warning = add_warning_to_error_config(None, "this is a warning")
    assert warning == {'error': 'this is a warning'}
    warning_2 = add_warning_to_error_config(warning, "another warning")
    assert warning_2 == {'error': 'another warning'}
    assert add_warning_to_error_config(warning_2, None) == {'error': ''}
