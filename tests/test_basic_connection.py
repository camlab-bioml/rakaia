import argparse
import pytest
from selenium.common import NoSuchElementException
from selenium.common.exceptions import SessionNotCreatedException
import socket
import platform
import os
from subprocess import Popen, PIPE
from rakaia.wsgi import cli_parser, main
from rakaia.entrypoint import init_app
from rakaia.app import init_dashboard
from rakaia.app import (
    init_pixel_level_callbacks,
    init_db_callbacks,
    init_roi_level_callbacks,
    init_object_level_callbacks)
import time
import signal
from flask import Flask
import dash_extensions
from dash_extensions.enrich import DashProxy
import dash_uploader as du
from conftest import skip_on
import tempfile


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true" or platform.system() != 'Linux',
                    reason="Only test the connection in a GA workflow due to passwordless sudo")
def test_for_connection():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    Popen(["echo yes| freeport 5000"], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    # p.stdin.write("yes\n")
    # p.communicate(input=b'yes')
    result = sock.connect_ex(('localhost', 5000))
    assert result != 0
    # assert result == 0
    new_process = Popen(["rakaia"],
                        stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    result = sock.connect_ex(('localhost', 5000))
    assert result == 103
    new_process.kill()
    result = sock.connect_ex(('localhost', 5000))
    assert result != 103 and result != 0


def recursive_wait_for_elem(app, elem, duration):
    if duration >= 2:
        return NoSuchElementException
    else:
        time.sleep(duration)
        try:
            app.find_element(elem).click()
        except NoSuchElementException:
            recursive_wait_for_elem(app, elem, int(1.1 * duration))


@skip_on(SessionNotCreatedException, "Selenium version is not compatible")
# @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true" or platform.system() != 'Linux',
#                     reason="Only test the connection in a GA workflow due to passwordless sudo")
def test_basic_app_load_from_locale(rakaia_flask_test_app, client):
    # credentials = base64.b64encode(b"rakaia_user:rakaia-1").decode('utf-8')
    assert str(type(rakaia_flask_test_app)) == "<class 'flask.app.Flask'>"
    response = client.get('/')
    assert response.status_code == 200
    overview_png = client.get('/static/assets/app-preview.png')
    assert overview_png.status_code == 200
    # test landing page alias
    assert client.get("/").data != b'Unauthorized Access'
    assert b'Scalable spatial biology analysis in the browser' in client.get("/").data
    # assert client.get('/', headers={"Authorization": "Basic {}".format(credentials)}).data == response.data
    # dash_duo.start_server(rakaia_flask_test_app.server)
    # for elem in ['#upload-image', '#tiff-image-type', '#image_layers', "#images_in_blend"]:
    #     assert dash_duo.find_element(elem) is not None
    # with pytest.raises(NoSuchElementException):
    #     assert dash_duo.find_element('#fake-input') is not None
    #
    # recursive_wait_for_elem(dash_duo, '#tab-quant', 1)
    #
    # for elem in ['#upload-quantification']:
    #     assert dash_duo.find_element(elem) is not None


def test_basic_cli_outputs():
    parser = cli_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args([])
    assert vars(args) == {'auto_open': True, 'is_dev_mode': False, 'loading': True, 'port': 5000, 'threading': True,
                          'use_local_dialog': False, 'persistence': True, 'swatches': None, 'array_type': 'float',
                          'serverside_overwrite': True, 'cache_dest': tempfile.gettempdir(), 'threads': 8}
    assert "rakaia can be initialized from the command line using:" in parser.usage
    parser = cli_parser()
    args = parser.parse_args(['-da'])
    assert vars(args) == {'auto_open': False, 'is_dev_mode': False, 'loading': True, 'port': 5000, 'threading': True,
                          'use_local_dialog': False, 'persistence': True, 'swatches': None, 'array_type': 'float',
                          'serverside_overwrite': True, 'cache_dest': tempfile.gettempdir(), 'threads': 8}
    assert "rakaia can be initialized from the command line using:" in parser.usage
    args = parser.parse_args(['-p', '8050'])
    assert vars(args) == {'auto_open': True, 'is_dev_mode': False, 'loading': True, 'port': 8050, 'threading': True,
                          'use_local_dialog': False, 'persistence': True, 'swatches': None, 'array_type': 'float',
                          'serverside_overwrite': True, 'cache_dest': tempfile.gettempdir(), 'threads': 8}
    assert "rakaia can be initialized from the command line using:" in parser.usage
    args = parser.parse_args(['-p', '8050', '-dp', '-at', 'int', '-dv'])
    assert vars(args) == {'auto_open': True, 'is_dev_mode': True, 'loading': True, 'port': 8050, 'threading': True,
                          'use_local_dialog': False, 'persistence': False, 'swatches': None, 'array_type': 'int',
                          'serverside_overwrite': True, 'cache_dest': tempfile.gettempdir(), 'threads': 8}

    with pytest.raises(SystemExit):
        parser.parse_args(['-v'])
    with pytest.raises(SystemExit):
        parser.parse_args(['-h'])
    with pytest.raises(SystemExit):
        parser.parse_args(['-t'])
    # with pytest.raises(SystemExit):
    #     main()


@pytest.mark.timeout(10)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true",
                    reason="Do not run main test in GA due to memory restrictions")
def test_basic_cli_outputs_main():
    """
    Assert that when valid arguments are passed to main, there is no system exit but rather the expected
    timeout after 5 seconds
    """

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(4)
    try:
        main(['-da'])
    except TimeoutException:
        assert True


def test_basic_app_return():
    config = {'auto_open': False, 'port': 5000, 'use_local_dialog': False, 'use_loading': False, 'persistence': True,
              'swatches': None, 'is_dev_mode': False, 'cache_dest': tempfile.gettempdir(), 'serverside_overwrite': True,
              'threads': 8}
    app = init_app(config)
    assert isinstance(app, Flask)
    app_2 = Flask("rakaia")
    dashboard = init_dashboard(app_2, "test_app", config)
    assert isinstance(dashboard, Flask)


def test_basic_callback_register():
    dash_app = DashProxy("fake_app")
    assert len(dash_app.callback_map) == 0
    du.configure_upload(dash_app, "/tmp/")
    app_config = {'use_local_dialog': False, 'use_loading': False, 'serverside_overwrite': True}
    init_pixel_level_callbacks(dash_app, "/tmp/", "test_app", app_config)
    init_object_level_callbacks(dash_app, "/tmp/", "test_app", app_config)
    init_db_callbacks(dash_app, "/tmp/", "test_app", app_config)
    init_roi_level_callbacks(dash_app, "/tmp/", "test_app", app_config)
    assert isinstance(dash_app, dash_extensions.enrich.DashProxy)
    assert dash_app.callback is not None
