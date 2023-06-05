import pytest
import os
from ccramic.app.entrypoint import init_app


@pytest.fixture(scope="module")
def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))


@pytest.fixture(scope="module")
def ccramic_flask_test_app():
    app = init_app()
    app.config.update({
        "TESTING": True,
    })
    yield app


@pytest.fixture(scope="module")
def client(ccramic_flask_test_app):
    return ccramic_flask_test_app.test_client()
