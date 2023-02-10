from subprocess import Popen, PIPE, STDOUT
import socket
import os

def test_for_connection():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # 2 Second Timeou
    p = Popen(["echo yes| freeport 8501"], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    print(p)
    # p.stdin.write("yes\n")
    # p.communicate(input=b'yes')
    result = sock.connect_ex(('localhost', 8501))
    assert result != 0
    # assert result == 0
    new_process = Popen(["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py"),
                    "--server.headless", "true"], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    result = sock.connect_ex(('localhost', 8501))
    assert result == 103
    new_process.kill()
    result = sock.connect_ex(('localhost', 8501))
    assert result != 103 and result != 0
    Popen(["echo yes| freeport 8501"], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)