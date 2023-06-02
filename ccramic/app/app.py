
from dash import dash_table
import tempfile
import dash_uploader as du
from flask_caching import Cache
from dash_extensions.enrich import DashProxy, html, dcc, \
    ServersideOutputTransform, FileSystemCache, ServersideBackend, FileSystemBackend
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import ctx, DiskcacheManager
import diskcache
from sqlite3 import DatabaseError
from .parsers import *
from .callbacks import init_callbacks
import shutil


def init_dashboard(server, authentic_id):

    with tempfile.TemporaryDirectory() as tmpdirname:
        # set the serveroutput cache dir and clean it every time a new app session is started
        # if whatever reason, the tmp is not writable, use a new directory as a backup
        if os.access("tmp", os.R_OK):
            cache_dest = os.path.join("tmp", authentic_id, "ccramic_cache")
        else:
            cache_dest = os.path.join(str(os.path.abspath(os.path.join(os.path.dirname(__file__)))), authentic_id,
                                      "ccramic_cache")
        if os.path.exists(cache_dest):
            shutil.rmtree(cache_dest)
        backend_dir = FileSystemBackend(cache_dir=cache_dest)
        dash_app = DashProxy(__name__,
                        transforms=[ServersideOutputTransform(backends=[backend_dir])],
                         external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                         server=server,
                         routes_pathname_prefix="/ccramic/")
        dash_app.title = "ccramic"
        server.config['APPLICATION_ROOT'] = "/ccramic"

        du.configure_upload(dash_app, tmpdirname)

    # VALID_USERNAME_PASSWORD_PAIRS = {
    #     'ccramic_user': 'ccramic'
    # }
    #
    # dash_auth.BasicAuth(
    #     dash_app,
    #     VALID_USERNAME_PASSWORD_PAIRS
    # )

    try:
        cache = Cache(dash_app.server, config={
            'CACHE_TYPE': 'redis',
            'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
        })
    except (ModuleNotFoundError, RuntimeError) as no_redis:
        try:
            cache = diskcache.Cache("./cache")
            background_callback_manager = DiskcacheManager(cache)
        except DatabaseError:
            cache = Cache(dash_app.server, config={
                'CACHE_TYPE': 'filesystem',
                'CACHE_DIR': 'cache-directory'
            })

    dash_app.layout = html.Div([
        # this modal is for the fullscreen view and does not belong in a nested tab
        dbc.Modal(children=dbc.ModalBody([dcc.Graph(config={"modeBarButtonsToAdd": [
                        # "drawline",
                        # "drawopenpath",
                        "drawclosedpath",
                        # "drawcircle",
                        "drawrect",
                        "eraseshape"],
                        'toImageButtonOptions': {'format': 'png', 'filename': 'canvas', 'scale': 1},
                        'edits': {'shapePosition': False}, 'scrollZoom': True}, relayoutData={'autosize': True},
                        id='annotation_canvas-fullscreen',
            style={"margin": "auto", "width": "100vw", "height": "100vh",
                   "max-width": "none", "max-height": "none"},
                        figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                              xaxis=go.XAxis(showticklabels=False),
                                              yaxis=go.YAxis(showticklabels=False))})]),
            id="fullscreen-canvas", fullscreen=True, size='xl',
        centered=True, style={"margin": "auto", "width": "100vw", "height": "100vh",
                              "max-width": "none", "max-height": "none"}),
        html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
        dbc.Tabs(id="all-tabs", children=[
            dbc.Tab(label='Image Annotation', tab_id='image-annotation', children=[
                html.Div([dbc.Tabs(id='pixel-level-analysis',
                children=[dbc.Tab(label='Pixel Analysis',
                tab_id='pixel-analysis',
                children=[html.Div([dbc.Row([dbc.Col(html.Div([
                        du.Upload(id='upload-image', max_file_size=10000,
                        max_total_size=10000, max_files=200,
                        filetypes=['png', 'tif', 'tiff', 'h5', 'mcd']),
                        dcc.Input(id="read-filepath", type="text",
                        placeholder="Add upload by file path (local runs only)", value=None),
                        dbc.Button("Add file by path", id="add-file-by-path",
                        className="mb-3", color="primary", n_clicks=0,
                        style={"margin-left": "20px", "margin-top": "10px"}),
                        html.Div([html.H5("Choose data collection", style={'width': '35%',
                                        'display': 'inline-block'}),
                        html.H5("Choose channel image", style={'width': '65%', 'display': 'inline-block'}),
                        dcc.Dropdown(id='data-collection', multi=False, options=[],
                        style={'width': '30%', 'display': 'inline-block', 'margin-right': '-30'}),
                        dcc.Dropdown(id='image_layers', multi=True,
                        style={'width': '70%', 'height': '100px', 'display': 'inline-block'})],
                        style={'width': '125%', 'height': '100%', 'display': 'inline-block', 'margin-left': '-30'}),
                        dcc.Slider(50, 100, 5, value=75, id='annotation-canvas-size'),
                        html.Div([html.H3("Image/Channel Blending", style={"margin-right": "50px",
                                                                           "margin-left": "30px"}),
                                  dbc.Button(children=html.Span([html.Div("Fullscreen"),
                                                                 html.I(className="fas fa-expand-arrows-alt",
                                                                        style={"display": "inline-block"}),
                                  ], style={"width": "100vw"}),
                                             id="make-canvas-fullscreen",
                                            color=None, n_clicks=0,
                                            style={"margin-left": "10px", "margin-top": "0px", "height": "100%"}),
                        html.Br()],
                        style={"display": "flex", "width": "100%"}),
                        dcc.Graph(config={"modeBarButtonsToAdd": [
                        # "drawline",
                        # "drawopenpath",
                        "drawclosedpath",
                        # "drawcircle",
                        "drawrect",
                        "eraseshape"],
                        'toImageButtonOptions': {'format': 'png', 'filename': 'canvas', 'scale': 1},
                            # disable scrollable zoom for now to control the scale bar
                        'edits': {'shapePosition': False}}, relayoutData={'autosize': True},
                        id='annotation_canvas',
                            style={"margin-top": "-30px"},
                            # style={"width": "100vw", "height": "100vh"},
                        figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                              xaxis=go.XAxis(showticklabels=False),
                                              yaxis=go.YAxis(showticklabels=False))}),
                    html.H6("Current canvas blend", style={'width': '75%'}),
                    html.Div(id='blend-color-legend', style={'whiteSpace': 'pre-line'}),
                    html.H6("Selection information", style={'width': '75%'}),
                    html.Div([dash_table.DataTable(id='selected-area-table',
                                                   columns=[{'id': p, 'name': p} for p in
                                                            ['Channel', 'Mean', 'Max', 'Min']],
                                                   data=None)], style={"width": "85%"})]),
                        width=9),
                        dbc.Col(html.Div([html.H5("Select channel to modify",
                                style={'width': '50%', 'display': 'inline-block'}),
                        html.Abbr("\u2753", title="Select a channel image to change colour or pixel intensity.",
                        style={'width': '5%', 'display': 'inline-block'}),
                        dcc.Dropdown(id='images_in_blend', multi=False),
                        html.Br(),
                        daq.ColorPicker(id="annotation-color-picker", label="Color Picker",
                        value=dict(hex="#1978B6")),
                        dcc.Graph(id="pixel-hist", figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                                         xaxis=go.XAxis(showticklabels=False),
                                                         yaxis=go.YAxis(showticklabels=False),
                                                        margin=dict(l=5, r=5, b=15, t=20, pad=0)),
                                                           },
                        style={'width': '60vh', 'height': '30vh', 'margin-left': '-30px'},
                        config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                        # keep zoom and pan bars to be able to modify the histogram view
                        # 'modeBarButtonsToRemove': ['zoom', 'pan']
                                },
                                  ),
                        html.Br(),
                        dcc.Checklist(options=[' apply/refresh filter'], value=[],
                        id="bool-apply-filter"),
                        dcc.Dropdown(['median', 'gaussian'], 'median', id='filter-type'),
                        dcc.Input(id="kernel-val-filter", type="number", value=3),
                        html.Br(),
                        html.Br(),
                        html.H6("Set custom scalebar value", style={'width': '75%'}),
                        dcc.Input(id="custom-scale-val", type="number", value=None),
                        html.Br(),
                        html.Br(),
                        dbc.Button("Create preset", id="preset-button", className="me-1",
                                          ),
                        dbc.Popover(dcc.Input(id="set-preset", type="text",
                        placeholder="Create a preset from the current channel", value=None),
                                    target="preset-button",
                                    trigger="hover",
                                    ),
                        html.Br(),
                        # daq.ToggleSwitch(label='use current preset',
                        #                  id='toggle-preset-use', labelPosition='bottom', persistence=True),
                        html.Br(),
                        dcc.Dropdown(options=[], value=None, id='preset-options'),
                        dbc.Tooltip(children="",
                                                      target="preset-options",
                                    id="hover-preset-information", trigger="hover"),
                        html.Br(),
                        html.Br(),

                        dbc.Button("Show download links", id="open-download-collapse", className="mb-3",
                        color="primary", n_clicks=0),
                        dbc.Tooltip(children="Open up the panel to get the download links.",
                                    target="open-download-collapse"),
                        html.Div(dbc.Collapse(
                        html.Div([html.A(id='download-link', children='Download File'),
                        html.Br(),
                        html.A(id='download-link-canvas-tiff', children='Download Canvas as tiff')]),
                        id="download-collapse", is_open=False), style={"minHeight": "100px"})]),
                        width=3)])])]),

            dbc.Tab(label="Image Gallery", tab_id='gallery-tab',
                        children=[html.Div([daq.ToggleSwitch(label='Change thumbnail on canvas zoom',
                        id='toggle-gallery-zoom', labelPosition='bottom', color="blue", style={"margin-right": "15px"}),
                                  daq.ToggleSwitch(label='View gallery by channel',
                                                   id='toggle-gallery-view', labelPosition='bottom', color="blue"),
                                            dcc.Dropdown(id='unique-channel-list', multi=False, options=[],
                                                         style={'width': '60%', 'display': 'inline-block',
                                                                'margin-right': '-30', 'margin-left': '15px'})
                                            ],
                                           style={"display": "flex"}),
                        html.Div(id="image-gallery", children=[
                        dbc.Row(id="image-gallery-row")]),
                                  ]),

            dbc.Tab(label="Panel Metadata", tab_id='metadata-tab', children=
                        [html.Div([dbc.Row([
                        dbc.Col(html.Div([
                        dash_table.DataTable(id='imc-metadata-editable', columns=[], data=None,
                                            editable=True)]), width=9),
                        dbc.Col(html.Div([du.Upload(id='upload-metadata', max_file_size=1000, max_files=1,
                                            filetypes=['csv'], upload_id="upload-image"),
                        html.Button("Download Edited metadata", id="btn-download-metadata"),
                        dcc.Download(id="download-edited-table")]),
                            width=3)])])])])
                          ])], id='tab-annotation'),
            dbc.Tab(tab_id='quantification-tab', label='Quantification/Clustering', children=[
                du.Upload(id='upload-quantification', max_file_size=5000, filetypes=['h5ad', 'h5'],
                    upload_id="upload-quantification"),
                html.Div([dbc.Row([
                    dbc.Col(html.Div(["Dimension Reduction/Clustering",
                                      dcc.Dropdown(id='dimension-reduction_options'),
                                      dcc.Graph(id='umap-plot', style={'width': '150vh', 'height': '150vh'})]),
                            width=6),
                    dbc.Col(html.Div(["Metadata Distribution",
                                      dcc.Dropdown(id='metadata_options'),
                                      dcc.Graph(id="metadata-distribution")]), width=6),
                ])]),

            ], id='tab-quant')
        ]),
        dcc.Loading(dcc.Store(id="uploaded_dict"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="session_config"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="hdf5_obj"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="blending_colours"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="image_presets"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="metadata_config"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="anndata"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="image-metadata"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="canvas-layers"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="alias-dict"), fullscreen=True, type="dot"),
        dcc.Loading(dcc.Store(id="static-session-var"), fullscreen=True, type="dot")
    ], style={"margin": "auto"})

    dash_app.enable_dev_tools(debug=True)

    init_callbacks(dash_app, tmpdirname, cache, authentic_id)

    return dash_app.server
