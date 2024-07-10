import pytest
import os
from rakaia.entrypoint import init_app
from functools import wraps
import tempfile

@pytest.fixture(scope="module")
def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))

@pytest.fixture(scope="module")
def rakaia_flask_test_app():
    app = init_app(cli_config={'use_local_dialog': False, 'use_loading': True,
                               'persistence': True, 'swatches': None, 'array_store_type': 'float',
                               'serverside_overwrite': True, 'is_dev_mode': True, 'cache_dest': tempfile.gettempdir()})
    app.config.update({
        "TESTING": True,
    })
    yield app


@pytest.fixture(scope="module")
def client(rakaia_flask_test_app):
    return rakaia_flask_test_app.test_client()

def skip_on(exception, reason="Default reason"):
    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(reason)

        return wrapper

    return decorator_func
