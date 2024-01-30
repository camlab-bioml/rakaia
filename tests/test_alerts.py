from ccramic.utils.alert import AlertMessage

def test_basic_alerts():
    alert_config = AlertMessage().warnings
    assert len(alert_config) == 12
