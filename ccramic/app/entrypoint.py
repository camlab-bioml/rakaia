from flask import Flask, redirect
from flask_caching import Cache
from flask import render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

_program = "ccramic"
__version__ = "0.6.0"


def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))


def init_app():
    # suppress numba depreciation warnings from umap
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    """Construct core Flask application with embedded Dash app."""
    # STATIC_DIR = os.path.dirname(os.path.join(get_current_dir(), "templates", "static"))
    app = Flask(__name__, instance_relative_config=False,
                static_url_path="", static_folder="static",
            template_folder="templates")

    # app.cache = Cache(app, config={'CACHE_TYPE': 'simple'})

    cache = Cache(config = {
        "DEBUG": True,  # some Flask specific configs
        "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
        "CACHE_DEFAULT_TIMEOUT": 300
    })
    cache.init_app(app)

    auth = HTTPBasicAuth()

    users = {
        "ccramic_user": generate_password_hash("ccramic-1")
    }

    app.config["APPLICATION_ROOT"] = "/"

    @auth.verify_password
    def verify_password(username, password):
        if username in users and \
                check_password_hash(users.get(username), password):
            return username

    @app.route('/')
    @auth.login_required
    def home():
        """Landing page."""
        return render_template(
            'home.html',
            title='ccramic',
            description='Cell-type Classification (using) Rapid Analysis (of) Multiplexed Imaging (mass) Cytometry.',
            template='home-template',
            body="This is a homepage served with Flask."
        )

    @app.route('/help/')
    @auth.login_required
    def help():
        """Landing page."""
        return render_template(
            'help.html')

    with app.app_context():
        # Import parts of our core Flask app

        # Import Dash application
        from .app import init_dashboard
        # use a unique uuid for the session id I/O
        authentic_user = str(uuid.uuid1())
        app = init_dashboard(app, authentic_user)
        # init_callbacks(app)

        return app
