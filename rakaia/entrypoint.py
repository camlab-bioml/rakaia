"""Module defining the main linking functions between the application dashboard and the
Flask application backend
"""

import uuid
import warnings
import os
from flask import Flask, render_template
from flask_caching import Cache
from flask_cors import CORS
# from flask_httpauth import HTTPBasicAuth
# from werkzeug.security import generate_password_hash, check_password_hash

def init_app(cli_config):
    """Initialize the parent Flask app that will wrap the Dash server.

    :param cli_config: dictionary of CLI app options from argparse
    :return: Parent Flask app object that will wrap the Dash Proxy server
    """
    # warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    # warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=DeprecationWarning)
    # Construct core Flask application with embedded Dash
    app = Flask(__name__, instance_relative_config=False,
                static_url_path="/static", static_folder="static",
                template_folder="templates")
    # dash.cache = Cache(dash, config={'CACHE_TYPE': 'simple'})
    CORS(app)
    app.config["SESSION_PERMANENT"] = False
    app.config['MIME_TYPES'] = {
        '.dzi': 'application/xml',
        '.xml': 'application/xml'
    }
    cache = Cache(config = {
        "DEBUG": cli_config['is_dev_mode'],  # some Flask specific configs
        "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
        "CACHE_DEFAULT_TIMEOUT": 300
    })
    cache.init_app(app)

    # auth = HTTPBasicAuth()
    #
    # users = {
    #     "rakaia_user": generate_password_hash("rakaia-1")
    # }

    app.config["APPLICATION_ROOT"] = "/"

    # set the steinbock mask dtype environment variable before the module is read
    # https://github.com/BodenmillerGroup/steinbock/issues/131
    os.environ["STEINBOCK_MASK_DTYPE"] = "uint32"

    # @auth.verify_password
    # def verify_password(username, password):
    #     if username in users and \
    #             check_password_hash(users.get(username), password):
    #         return username
    #     return None

    @app.route('/')
    # @auth.login_required
    def home():
        """Landing page."""
        return render_template(
            'home.html',
            title='rakaia',
            description='Analyze multiplexed imaging datasets interactively and quickly.',
            template='home-template',
            body="This is a homepage served with Flask."
        )

    @app.route('/help/')
    # @auth.login_required
    def help_landing():
        """Landing page."""
        return render_template(
            'help.html')

    with app.app_context():
        # Import parts of our core Flask dash
        # Import Dash application
        from rakaia.app import init_dashboard
        # use a unique uuid for the session id I/O
        authentic_user = str(uuid.uuid1())
        app = init_dashboard(app, authentic_user, config=cli_config)
        # init_callbacks(dash)
        return app
