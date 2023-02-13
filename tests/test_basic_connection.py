from subprocess import Popen, PIPE, STDOUT
import socket
import os
import pytest
import platform


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true" or platform.system() != 'Linux',
                    reason="Only test the connection in a GA workflow due to passwordless sudo")
def test_for_connection():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    Popen(["echo yes| freeport 8501"], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    # p.stdin.write("yes\n")
    # p.communicate(input=b'yes')
    result = sock.connect_ex(('localhost', 8501))
    assert result != 0
    # assert result == 0
    new_process = Popen(["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py"),
                         "--server.headless", "true", "--server.port", "8501"],
                        stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    result = sock.connect_ex(('localhost', 8501))
    assert result == 103
    new_process.kill()
    result = sock.connect_ex(('localhost', 8501))
    assert result != 103 and result != 0
