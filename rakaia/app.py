import tempfile
import shutil
import os
import dash_uploader as du
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, FileSystemBackend
import dash_bootstrap_components as dbc

from rakaia.callbacks.metadata import init_metadata_level_callbacks
from rakaia.components.layout import register_app_layout
from rakaia.callbacks.pixel import init_pixel_level_callbacks
from rakaia.callbacks.object import init_object_level_callbacks
from rakaia.callbacks.roi import init_roi_level_callbacks
from rakaia.callbacks.db import init_db_callbacks

def init_dashboard(server, authentic_id, config=None):
    """Initialize the dashboard server.

    :param server: The parent Flask app for rakaia to run
    :param authentic_id: A uuid generated on CLI initialization
    :param config: dictionary of CLI app options from argparse
    :return: Dash proxy server object
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # set the server output cache dir and clean it every time a new dash session is started
        # if whatever reason, the tmp is not writable, use a new directory as a backup
        # do not try to clear the caches if not using overwrite, for example deployed to a shared server
        if os.access(config['cache_dest'], os.R_OK) and config['serverside_overwrite']:
            # cleaning the tmp dir for any sub-directory that has rakaia cache in it
            cache_subdirs = [x[0] for x in os.walk(config['cache_dest']) if 'rakaia_cache' in x[0]]
            # remove any parent directory that has a rakaia cache in it
            for cache_dir in cache_subdirs:
                if os.access(os.path.dirname(cache_dir), os.R_OK) and os.access(cache_dir, os.R_OK) and \
                        os.path.isdir(cache_dir):
                    shutil.rmtree(os.path.dirname(cache_dir))
            cache_dest = os.path.join(config['cache_dest'], authentic_id, "rakaia_cache")
        else:
            # try to find the default tempdir if the supplied cache is not permissible
            cache_dest = os.path.join(str(tempfile.gettempdir()), authentic_id, "rakaia_cache")

        if os.path.exists(cache_dest):
            shutil.rmtree(cache_dest, ignore_errors=True)

        backend_dir = FileSystemBackend(cache_dir=cache_dest)
        dash_app = DashProxy(__name__,
                        update_title=None,
                        transforms=[ServersideOutputTransform(backends=[backend_dir])],
                        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                        server=server,
                        routes_pathname_prefix="/rakaia/",
                        suppress_callback_exceptions=True,
                        prevent_initial_callbacks=True)
        dash_app._favicon = 'rakaia.ico'
        dash_app.title = "rakaia"
        server.config['APPLICATION_ROOT'] = "/rakaia"
        # do not use debugging mode if production is used
        server.config['FLASK_DEBUG'] = config['is_dev_mode']

        # Can configure a custom http request handler for public instances
        # https://github.com/fohrloop/dash-uploader/blob/dev/docs/dash-uploader.md#4-custom-handling-of-http-requests
        du.configure_upload(dash_app, cache_dest)

    # for now, do not initiate the dash caching as it interferes on Windows OS and isn't strictly
    # useful when serverside components can cache large stores much more effectively

    # VALID_USERNAME_PASSWORD_PAIRS = {
    #     'rakaia_user': 'rakaia'
    # }
    #
    # dash_auth.BasicAuth(
    #     dash_app,
    #     VALID_USERNAME_PASSWORD_PAIRS
    # )

    # try:
    #     cache = Cache(dash_app.server, config={
    #         'CACHE_TYPE': 'redis',
    #         'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    #     })
    # except (ModuleNotFoundError, RuntimeError) as no_redis:
    #     try:
    #         cache = diskcache.Cache(os.path.join("/tmp/", "diskcache"))
    #         background_callback_manager = DiskcacheManager(cache)
    #     except DatabaseError:
    #         cache = Cache(dash_app.server, config={
    #             'CACHE_TYPE': 'filesystem',
    #             'CACHE_DIR': 'cache-directory'
    #         })
    #
    # cache = diskcache.Cache(os.path.join("/tmp/", "diskcache"))
    # background_callback_manager = DiskcacheManager(cache)

    dash_app.layout = register_app_layout(config, cache_dest)

    dash_app.enable_dev_tools(debug=config['is_dev_mode'])

    init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_object_level_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_roi_level_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_db_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_metadata_level_callbacks(dash_app, tmpdirname, authentic_id, config)

    return dash_app.server
