import base64

import pytest
from selenium.common import NoSuchElementException
import time
import socket
import platform
import os
from subprocess import Popen, PIPE


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
    new_process = Popen(["ccramic"],
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
            recursive_wait_for_elem(app, elem, int(1.1*duration))


def test_basic_app_load_from_locale(ccramic_flask_test_app, client):
    credentials = base64.b64encode(b"ccramic_user:ccramic-1").decode('utf-8')
    assert str(type(ccramic_flask_test_app)) == "<class 'flask.app.Flask'>"
    response = client.get('/', headers={"Authorization": "Basic {}".format(credentials)})
    assert response.status_code == 200
    # test landing page alias
    assert client.get("/").data == b'Unauthorized Access'
    client.get('/', headers={"Authorization": "Basic {}".format(credentials)}).data == response.data
    # for elem in ['#upload-image', '#tiff-image-type', '#image_layers', "#images_in_blend"]:
    #     assert dash_duo.find_element(elem) is not None
    # with pytest.raises(NoSuchElementException):
    #     assert dash_duo.find_element('#fake-input') is not None
    #
    # recursive_wait_for_elem(dash_duo, '#tab-quant', 1)
    #
    # for elem in ['#upload-quantification']:
    #     assert dash_duo.find_element(elem) is not None
