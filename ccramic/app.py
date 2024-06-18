
import tempfile
import dash_uploader as du
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, FileSystemBackend
from ccramic.callbacks.pixel_level_callbacks import init_pixel_level_callbacks
from ccramic.callbacks.cell_level_callbacks import init_cell_level_callbacks
from ccramic.callbacks.roi_level_callbacks import init_roi_level_callbacks
from ccramic.callbacks.db_callbacks import init_db_callbacks
import shutil
import os
import dash_bootstrap_components as dbc
from ccramic.components.layout import register_app_layout

def init_dashboard(server, authentic_id, config=None):

    with tempfile.TemporaryDirectory() as tmpdirname:
        # set the server output cache dir and clean it every time a new dash session is started
        # if whatever reason, the tmp is not writable, use a new directory as a backup
        if os.access(config['cache_dest'], os.R_OK):
            # TODO: establish cleaning the tmp dir for any sub directory that has ccramic cache in it
            cache_subdirs = [x[0] for x in os.walk(config['cache_dest']) if 'ccramic_cache' in x[0]]
            # remove any parent directory that has a ccramic cache in it
            for cache_dir in cache_subdirs:
                if os.access(os.path.dirname(cache_dir), os.R_OK) and os.access(cache_dir, os.R_OK):
                    shutil.rmtree(os.path.dirname(cache_dir))
            cache_dest = os.path.join(config['cache_dest'], authentic_id, "ccramic_cache")
        else:
            # try to find the default tempdir if the supplied cache is not permissible
            cache_dest = os.path.join(str(tempfile.gettempdir()), authentic_id, "ccramic_cache")

        if os.path.exists(cache_dest):
            shutil.rmtree(cache_dest)

        backend_dir = FileSystemBackend(cache_dir=cache_dest)
        dash_app = DashProxy(__name__,
                        update_title=None,
                        transforms=[ServersideOutputTransform(backends=[backend_dir])],
                        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                        server=server,
                        routes_pathname_prefix="/ccramic/",
                        suppress_callback_exceptions=True,
                        prevent_initial_callbacks=True)
        dash_app._favicon = 'ccramic.ico'
        dash_app.title = "ccramic"
        server.config['APPLICATION_ROOT'] = "/ccramic"
        # do not use debugging mode if production is used
        server.config['FLASK_DEBUG'] = config['is_dev_mode']

        # TODO: configure a custom http request handler for public instances
        # https://github.com/fohrloop/dash-uploader/blob/dev/docs/dash-uploader.md#4-custom-handling-of-http-requests
        du.configure_upload(dash_app, cache_dest)

    #for now, do not initiate the dash caching as it interferes on Windows OS and isn't strictly
    # useful when serverside components can cache large stores much more effectively

    # VALID_USERNAME_PASSWORD_PAIRS = {
    #     'ccramic_user': 'ccramic'
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
    init_cell_level_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_roi_level_callbacks(dash_app, tmpdirname, authentic_id, config)
    init_db_callbacks(dash_app, tmpdirname, authentic_id, config)

    return dash_app.server
