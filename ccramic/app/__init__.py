import ccramic.app.routes
from flask import Flask
from flask_caching import Cache
from flask import render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid

_program = "ccramic"
__version__ = "0.1.0"


def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))


def init_app():
    """Construct core Flask application with embedded Dash app."""
    # STATIC_DIR = os.path.dirname(os.path.join(get_current_dir(), "templates", "static"))
    app = Flask(__name__, instance_relative_config=False,
                static_url_path="", static_folder="static",
            template_folder="templates")

    app.cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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
    @app.route('/ccramic')
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
        import ccramic.app.routes

        # Import Dash application
        from .app import init_dashboard, init_callbacks
        # use a unique uuid for the session id I/O
        authentic_user = str(uuid.uuid1())
        app = init_dashboard(app, authentic_user)
        # init_callbacks(app)

        return app
