_program = "ccramic"
__version__ = "0.1.0"
import ccramic.routes
from flask import Flask
from flask_caching import Cache
from flask import render_template


def init_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
    # app.config.from_object('config.Config')

    app.cache = Cache(app, config={'CACHE_TYPE': 'simple'})

    @app.route('/')
    @app.route('/ccramic')
    def home():
        """Landing page."""
        return render_template(
            'index.jinja2',
            title='ccramic',
            description='Cell-type Classification (using) Rapid Analysis (of) Multiplexed Imaging (mass) Cytometry.',
            template='home-template',
            body="This is a homepage served with Flask."
        )

    with app.app_context():
        # Import parts of our core Flask app
        import ccramic.routes

        # Import Dash application
        from ccramic.app.app import init_dashboard, init_callbacks
        app = init_dashboard(app)
        # init_callbacks(app)

        return app
