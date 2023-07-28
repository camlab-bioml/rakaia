
from dash import dash_table
import tempfile
import dash_uploader as du
from dash_extensions.enrich import DashProxy, html, ServersideOutputTransform, FileSystemBackend
import dash_daq as daq
import dash_bootstrap_components as dbc
from .callbacks.pixel_level_callbacks import init_pixel_level_callbacks
from .callbacks.cell_level_callbacks import init_cell_level_callbacks
from .inputs.pixel_level_inputs import *
import shutil
import os
def init_dashboard(server, authentic_id):

    with tempfile.TemporaryDirectory() as tmpdirname:
        # set the serveroutput cache dir and clean it every time a new app session is started
        # if whatever reason, the tmp is not writable, use a new directory as a backup
        if os.access("/tmp/", os.R_OK):
            cache_dest = os.path.join("/tmp/", authentic_id, "ccramic_cache")
        else:
            cache_dest = os.path.join(str(os.path.abspath(os.path.join(os.path.dirname(__file__)))), authentic_id,
                                      "ccramic_cache")
        if os.path.exists(cache_dest):
            shutil.rmtree(cache_dest)
        backend_dir = FileSystemBackend(cache_dir=cache_dest)
        dash_app = DashProxy(__name__,
                             update_title=None,
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

    dash_app.layout = html.Div([
        # this is the generic error modal that will pop up on specific errors return by the alert dict
        dbc.Modal(children=dbc.ModalBody([html.Div(id='alert-information', style={'whiteSpace': 'pre-line'})]),
                  id="alert-modal"),
        # this modal is for the fullscreen view and does not belong in a nested tab
        dbc.Modal(children=dbc.ModalBody([render_default_annotation_canvas(input_id="annotation_canvas-fullscreen",
                                                                           fullscreen_mode=True)]),
            id="fullscreen-canvas", fullscreen=True, size='xl',
        centered=True, style={"margin": "auto", "width": "100vw", "height": "100vh",
                              "max-width": "none", "max-height": "none"}),
        # modal for the dataset information
        dbc.Modal(children=dbc.ModalBody([dash_table.DataTable(id='dataset-preview-table', columns=[], data=None,
                                            editable=False, filter_action='native')]),
                  id="dataset-preview", size='xl'),
        html.H2("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry)"),
        dbc.Tabs(id="all-tabs", children=[
            dbc.Tab(label='Image Annotation', tab_id='image-annotation', children=[
                html.Div([dbc.Tabs(id='pixel-level-analysis',
                children=[dbc.Tab(label='Pixel Analysis',
                tab_id='pixel-analysis',
                children=[html.Div([dbc.Row([dbc.Col(html.Div([
                        du.Upload(id='upload-image', max_file_size=30000,
                                  text='Import imaging data from MCD or tiff files using drag and drop',
                                  chunk_size=100,
                        max_total_size=30000, max_files=200,
                        filetypes=['png', 'tif', 'tiff', 'h5', 'mcd', 'txt'], default_style={"margin-top": "20px",
                                                                                             "height": "10vh"}),
                        dcc.Input(id="read-filepath", type="text",
                        placeholder="Import imaging file using filepath (local runs only)", value=None,
                                  style={"width": "85%"}),
                        dbc.Button("Add file by path", id="add-file-by-path",
                        className="mb-3", color="primary", n_clicks=0,
                        style={"margin-left": "20px", "margin-top": "10px"}),
                        html.Div([html.Span([html.H5("Choose data collection", style={'width': '35%',
                                        'display': 'inline-block'}),
                                             dbc.Button("Dataset info", id="show-dataset-info",
                        className="mb-3", color="primary", n_clicks=0,
                        style={"margin-left": "-235px", "margin-top": "10px"}),
                                             dbc.Button(children=html.Span([html.Abbr(html.I(className="fa fa-trash",
                                                                                   style={"display": "iflex"}),
                                                title="Remove the current data collection. "
                                                      "(IMPORTANT): cannot be undone."),
                                                                            ], style={"width": "100vw"}),
                                                        id="remove-collection",
                                                        color=None, n_clicks=0,
                                                        style={"margin-top": "-5px", "height": "75%"}),
                                             ], style={"width": "50%", "margin-right": "25px"}),
                        html.H5("Choose channel image", style={'width': '65%', 'display': 'inline-block',
                                                               'margin-left': '15px'}),
                        dcc.Dropdown(id='data-collection', multi=False, options=[],
                        style={'width': '50%', 'display': 'inline-block', 'margin-right': '-50'}),
                        dcc.Dropdown(id='image_layers', multi=True,
                        style={'width': '65%', 'height': '100px', 'display': 'inline-block', 'margin-left': '220px',
                               'margin-top': '-22.5px'})],
                        style={'width': '125%', 'height': '100%', 'display': 'inline-block'}),
                        dcc.Slider(50, 100, 5, value=75, id='annotation-canvas-size'),
                        html.Div([html.H3("", style={"margin-right": "50px",
                                                                           "margin-left": "30px"}),
                                  dbc.Button(children=html.Span([html.Div("Fullscreen"),
                                                                 html.I(className="fas fa-expand-arrows-alt",
                                                                        style={"display": "inline-block"}),
                                  ], style={"width": "100vw", "margin-top": "-5px", "margin-bottom": "10px"}),
                                             id="make-canvas-fullscreen",
                                            color=None, n_clicks=0,
                                            style={"margin-left": "10px", "margin-top": "0px", "height": "100%"}),
                                  dbc.Button("Auto-fit canvas", id="autosize-canvas",
                                             style={"margin-left": "10px", "margin-top": "5px", "height": "100%"}),
                                  html.Div(style={"margin-left": "20px", "margin-right": "10px",
                                                                    "margin-top": "10px", "height": "100%",
                                                  "width": "150%"},
                                          id="bound-shower"),
                                  dcc.Input(id="set-x-auto-bound", type="number", value=None,
                                            placeholder="Set x-coord",
                                            style={"margin-left": "10px",
                                                   "margin-top": "10px", "height": "100%", "width": "15vh"}
                                            ),
                                  dcc.Input(id="set-y-auto-bound", type="number", value=None,
                                            placeholder="Set y-coord",
                                            style={"margin-left": "10px",
                                                   "margin-top": "10px", "height": "100%", "width": "15vh"}
                                            ),
                                  dbc.Button("Set", id="activate-coord",
                                             style={"width": "50px", "height": "35px", "margin-left": "15px",
                                                    "margin-right": "10px", "margin-top": "7px"}),
                        html.Br()],
                        style={"display": "flex", "width": "100%", "margin-bottom": "15px"}),
                        html.Div([render_default_annotation_canvas(input_id="annotation_canvas")],
                                 style={"margin-top": "-22px"}, id="canvas-div-holder"),
                    html.H6("Current canvas blend", style={'width': '75%'}),
                    html.Div(id='blend-color-legend', style={'whiteSpace': 'pre-line'}),
                    dbc.Button("Add region annotation", id="region-annotation",
                               style={"margin-top": "5px", "height": "100%"},
                               disabled=True),
                    html.Br(),
                    html.Br(),
                    dbc.Button("Show/hide region statistics", id="compute-region-statistics", className="mb-3",
                               color="primary", n_clicks=0),
                    html.Br(),
                    html.Div(dbc.Collapse(
                        html.Div([html.H6("Selection information", style={'width': '75%'}),
                                  html.Div([dash_table.DataTable(id='selected-area-table',
                                                                 columns=[{'id': p, 'name': p} for p in
                                                                          ['Channel', 'Mean', 'Max', 'Min']],
                                                                 data=None)], style={"width": "85%"})
                                  ]),
                        id="area-stats-collapse", is_open=False), style={"minHeight": "100px"})]),
                        width=9),
                        dbc.Col(html.Div([html.H5("Select channel to modify",
                                style={'width': '50%', 'display': 'inline-block'}),
                        html.Abbr("\u2753", title="Select a channel in the current blend to \nchange colour, "
                                                  "pixel intensity, or apply a filter.",
                        style={'width': '5%', 'display': 'inline-block'}),
                        dcc.Dropdown(id='images_in_blend', multi=False),
                        html.Br(),
                        daq.ColorPicker(id="annotation-color-picker", label="Current channel color",
                        value=dict(hex="#00ABFC", rgb=None)),
                        dcc.Loading(
                        dcc.Graph(id="pixel-hist", figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                                         xaxis=go.XAxis(showticklabels=False),
                                                         yaxis=go.YAxis(showticklabels=False),
                                                        margin=dict(l=5, r=5, b=15, t=20, pad=0)),
                                                           },
                        style={'width': '60vh', 'height': '30vh', 'margin-left': '-30px'},
                        # config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                        # keep zoom and pan bars to be able to modify the histogram view
                        # 'modeBarButtonsToRemove': ['zoom', 'pan']
                                #},
                                  ),
                            type="default", fullscreen=False),
                        html.Br(),
                        html.Div([dcc.RangeSlider(0, 100, 1, value=[None, None], marks=dict([(i,str(i)) for \
                                                                                        i in range(0, 100, 25)]),
                                                  id='pixel-intensity-slider',
                                                  tooltip={"placement": "top", "always_visible": True})],
                                        style={"width": "92.5%", "margin-left": "27px", "margin-top": "-50px"}),
                        html.Br(),
                        html.H6("Import mask"),
                        dbc.Modal(children=dbc.ModalBody(
                        [html.H6("Set the label for the imported mask"),
                                 html.Div([dcc.Input(id="input-mask-name", type="text", value=None,
                                                     style={"width": "65%", "margin-right": "10px", "height": "50%"}),
                         daq.ToggleSwitch(label='Derive cell boundary', id='derive-cell-boundary',
                                          labelPosition='bottom', color="blue", value=True,
                                          style={"margin-right": "-30px", "margin-left": "10px"})],
                                          style={"display": "flex"}),
                         dbc.Button("Set mask import", id="set-mask-name", className="me-1")]),
                                                    id="mask-name-modal", size='l',
                            style={"margin-left": "10px", "margin-top": "15px"}),
                        du.Upload(id='upload-mask', max_file_size=30000,
                                  text='Import mask in tiff format using drag and drop',
                                                    max_total_size=30000, max_files=1,
                                                    chunk_size=100,
                                                    filetypes=['tif', 'tiff'],
                                                    default_style={"margin-top": "20px", "height": "3.5vh"}),
                        html.Br(),
                        html.Div([dcc.Loading(dcc.Dropdown(id='mask-options', multi=False, options=[],
                                                       style={'width': '100%', 'display': 'inline-block',
                                                    'margin-right': '-50'}), type="default", fullscreen=False),
                        dcc.Slider(0, 100, 2.5, value=100, id='mask-blending-slider', marks={0: '0%', 25: '25%',
                                                                                           50: '50%', 75: '75%', 100:
                                                                                           '100%'}),
                        html.Div([daq.ToggleSwitch(label='Apply mask',id='apply-mask', labelPosition='bottom',
                                                           color="blue", style={"margin-left": "60px"}),
                                  html.Abbr(dcc.Checklist(options=[' add boundary'], value=[],
                                                id="add-mask-boundary", style={"margin-left": "35px",
                                                                               "margin-top": "10px"}),
                                            title="Use this feature only if the cell boundary was not "
                                                  "derived on import"),
                                  ], style={"display": "flex"})]),

                        html.Br(),
                        dcc.Checklist(options=[' apply/refresh filter'], value=[],
                                                        id="bool-apply-filter"),
                        dcc.Dropdown(['median', 'gaussian'], 'median', id='filter-type'),
                        dcc.Input(id="kernel-val-filter", type="number", value=3),
                        html.Br(),
                        html.Br(),
                        html.Div([daq.ToggleSwitch(label='Toggle legend',
                                                   id='toggle-canvas-annotations', labelPosition='bottom',
                                                   value=True, color="blue", style={"width": "75%",
                                                                                "margin-left": "-15px"}),
                                  html.Div([html.H6("Set custom scalebar value", style={'width': '110%'}),
                                            dcc.Input(id="custom-scale-val", type="number", value=None,
                                                      style={"width": "60%", "margin-left": "30px"})],
                                           style={"display": "block"})],
                                 style={"display": "flex"}),
                        html.Br(),
                        html.Br(),
                        dcc.Checklist(options=[' show channel intensities on hover'],
                                      value=[], id="channel-intensity-hover"),
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
                        html.Div([html.A(id='download-link', children='Download current session'),
                        html.Br(),
                        html.A(id='download-link-canvas-tiff', children='Download Canvas as tiff')]),
                        id="download-collapse", is_open=False), style={"minHeight": "100px"})]),
                        width=3)])])]),

            dbc.Tab(label="Image Gallery", tab_id='gallery-tab',
                        children=[html.Div([daq.ToggleSwitch(label='Change thumbnail on zoom',
                        id='toggle-gallery-zoom', labelPosition='bottom', color="blue", style={"margin-right": "15px",
                                                                                               "margin-top": "10px"}),
                                  daq.ToggleSwitch(label='View gallery by channel',
                                                   id='toggle-gallery-view', labelPosition='bottom', color="blue",
                                                   style={"margin-right": "7px", "margin-top": "10px"}
                                                   ),
                                    daq.ToggleSwitch(label='Use default scaling for preview',
                                                     value=True,
                                                             id='default-scaling-gallery', labelPosition='bottom',
                                                             color="blue", style={"margin-left": "15px",
                                                                                  "margin-top": "10px"}),
                                            dcc.Dropdown(id='unique-channel-list', multi=False, options=[],
                                                         style={'width': '60%', 'display': 'inline-block',
                                                                'margin-right': '-30', 'margin-left': '15px',
                                                                "margin-top": "10px"})
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
                                            text='Import panel metadata in CSV format using drag and drop',
                                            filetypes=['csv'], upload_id="upload-image"),
                        html.Button("Download Edited metadata", id="btn-download-metadata"),
                        dcc.Download(id="download-edited-table")]),
                            width=3)])])])])
                          ])], id='tab-annotation'),
            dbc.Tab(tab_id='quantification-tab', label='Quantification/Clustering', children=[
                du.Upload(id='upload-quantification', max_file_size=5000, filetypes=['h5ad', 'h5', 'csv'],
                          text='Import cell quantification results in CSV format using drag and drop',
                          max_files=1, upload_id="upload-quantification"),
                html.Div([dbc.Row([
                    dbc.Col(html.Div([html.Br(),
                        html.H6("Cell-Level Marker Expression"),
                                      dcc.RadioItems(['max', 'mean', 'min'], 'mean',
                                                     inline=True, id="quantification-bar-mode"),
                                      dcc.Graph(id="quantification-bar-full",
                                                figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                                                       xaxis=go.XAxis(showticklabels=False),
                                                                       yaxis=go.YAxis(showticklabels=False),
                                                                       margin=dict(l=5, r=5, b=15, t=20, pad=0)),
                                                        })]), width=6),
                    dbc.Col(html.Div([html.Br(),
                                      html.H6("Dimension Reduction"),
                                      dcc.Loading(dcc.Dropdown(id='umap-projection-options', multi=False, options=[]),
                                                  type="default", fullscreen=False),
                                      dcc.Graph(id="umap-plot",
                                                figure={'layout': dict(xaxis_showgrid=False, yaxis_showgrid=False,
                                                                       xaxis=go.XAxis(showticklabels=False),
                                                                       yaxis=go.YAxis(showticklabels=False),
                                                                       margin=dict(l=5, r=5, b=15, t=20, pad=0)),
                                                        })]), width=6)
                ])]),

            ], id='tab-quant')
        ]),
        dcc.Loading(dcc.Store(id="uploaded_dict"), type="default", fullscreen=True),
        # use a blank template for the lazy loading
        dcc.Loading(dcc.Store(id="uploaded_dict_template"), type="default", fullscreen=True),
        dcc.Store(id="session_config"),
        dcc.Store(id="window_config"),
        dcc.Store(id="param_config"),
        dcc.Store(id="session_alert_config"),
        dcc.Store(id="hdf5_obj"),
        dcc.Store(id="blending_colours"),
        dcc.Store(id="image_presets"),
        dcc.Store(id="metadata_config"),
        dcc.Store(id="anndata"),
        dcc.Store(id="image-metadata"),
        dcc.Store(id="canvas-layers"),
        dcc.Store(id="alias-dict"),
        dcc.Store(id="static-session-var"),
        dcc.Store(id="session_config_quantification"),
        dcc.Store(id="quantification-dict"),
        dcc.Store(id="mask-dict"),
        dcc.Store(id="mask-uploads"),
        dcc.Store(id="figure-cache"),
        dcc.Store(id="uploads"),
        dcc.Store(id="current_canvas_image"),
        dcc.Store(id="umap-projection"),
        dcc.Store(id="annotations-dict"),
    ], style={"margin": "17.5px"})

    dash_app.enable_dev_tools(debug=True)

    init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id)
    init_cell_level_callbacks(dash_app)

    return dash_app.server
