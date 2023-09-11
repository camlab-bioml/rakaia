
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
# from sd_material_ui import AutoComplete
import dash_ag_grid as dag
import dash_mantine_components as dmc
from plotly.graph_objs.layout import YAxis, XAxis
from .entrypoint import __version__
def init_dashboard(server, authentic_id, config=None):

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
        html.Header(
            className="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow",
            children=[
                html.A("ccramic", className="navbar-brand me-0 px-3", href="#"),
                html.A(f"v{__version__}", className="navbar-brand me-0 px-3", href="#", style={"float": "right"})],
            style={"margin-bottom": "15px"}),
            dbc.Tab(label='Image Annotation', tab_id='image-annotation', active_label_style={"color": "#FB79B3"},
                    children=[
                html.Div([dbc.Tabs(id='pixel-level-analysis',
                children=[dbc.Tab(# label_class_name="fa-regular fa-file-image",
                                  label="Image analysis",
                                  # label_style={"text-transform": "capitalize", "font-weight": "normal"},
                tab_id='pixel-analysis',
                children=[
                    dbc.Offcanvas(
                        id="inputs-offcanvas",
                        title="Configure inputs for ccramic",
                        is_open=True,
                        children=[html.H5("Import images", style={'width': '65%',
                                                                     }),
                                  du.Upload(id='upload-image', max_file_size=30000,
                                  text='Import imaging data from MCD or tiff files using drag and drop',
                                  chunk_size=100,
                        max_total_size=30000, max_files=200,
                        filetypes=['png', 'tif', 'tiff', 'h5', 'mcd', 'txt'], default_style={"margin-top": "20px",
                                                                                             "height": "10vh"}),
                        html.Br(),
                        dcc.Input(id="read-filepath", type="text",
                        placeholder="Import imaging file using filepath (local runs only)",
                        value=None, style={"width": "100%", "height": "10%"}),
                        dbc.Button("Add file by path", id="add-file-by-path",
                                   className="mb-3", color="primary", n_clicks=0, style={"margin-top": "10px"}),
                        add_local_file_dialog(use_local_dialog=config['use_local_dialog']),
                        dbc.Tooltip("Browse the local file system using a dialog."
                                " IMPORTANT: may not be compatible with the specific OS.", target="local-dialog-file"),
                        html.Div([html.Span([
                            dbc.Button(children=html.Span([html.I(className="fa-solid fa-circle-info",
                            style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                            html.Div("Dataset info")], style={"display": "flex"}), id="show-dataset-info",
                            className="mb-3", color="primary", n_clicks=0, style={"margin-top": "10px"}),
                            dbc.Button(children=html.Span([html.Abbr(html.I(className="fa fa-trash",
                            style={"display": "iflex"}))], style={"width": "100vw"}),
                            id="remove-collection", color=None, n_clicks=0,
                            style={"margin-top": "-5px", "height": "10%"}),
                            dbc.Tooltip("Remove the current data collection. "
                            "(IMPORTANT): cannot be undone.", target="remove-collection")], style={"width": "100%"}),
                            html.Br(),
                            html.Br(),
                            html.H5("Choose data collection/ROI", style={'width': '65%',
                                                                     }),
                            dcc.Dropdown(id='data-collection', multi=False, options=[],
                                         style={'width': '100%'}),
                            html.Br(),
                            html.H5("Import mask"),
                            dbc.Modal(children=dbc.ModalBody(
                                [html.H6("Set the label for the imported mask"),
                                 html.Div([dcc.Input(id="input-mask-name", type="text", value=None,
                                                     style={"width": "65%", "margin-right": "10px", "height": "50%"}),
                                           daq.ToggleSwitch(label='Derive cell boundary', id='derive-cell-boundary',
                                                            labelPosition='bottom', color="blue", value=False,
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
                                      default_style={"margin-top": "20px", "height": "10%"}),
                            html.Br(),
                            html.H5("Import quantification results"),
                            du.Upload(id='upload-quantification', max_file_size=5000,
                                      filetypes=['h5ad', 'h5', 'csv'],
                                      text='Import cell quantification results in CSV format using drag and drop',
                                      max_files=1, upload_id="upload-quantification",
                                      default_style={"margin-top": "20px", "height": "10%"}),
                            html.Br(),
                            html.H5("Downloads"),
                            dbc.Button(children=html.Span([html.I(className="fa-solid fa-download",
                                                                  style={"display": "inline-block",
                                                                         "margin-right": "7.5px", "margin-top": "3px"}),
                                                           html.Div("Show download links")], style={"display": "flex"}),
                                       id="open-download-collapse", className="mx-auto", color=None, n_clicks=0,
                                       style={"margin-top": "10px"}),
                            dbc.Tooltip(children="Open up the panel to get the download links.",
                                        target="open-download-collapse"),
                            html.Div(dbc.Collapse(
                                dcc.Loading(html.Div([html.A(id='download-link', children='Download current session'),
                                                      html.Br(),
                                                      html.A(id='download-link-canvas-tiff',
                                                             children='Download Canvas as tiff (no annotations)'),
                                                      html.Br(),
                                                      html.A(id='download-canvas-interactive-html',
                                                             children='Download Canvas as as interactive HTML')
                                                      ]),
                                            fullscreen=False, type="default"),
                                id="download-collapse", is_open=False), style={"minHeight": "100px"})
                        ],
                        style={'width': '100%', 'height': '100%', "margin-top": "5px"}
                                 )], style={"width": "33%", "padding": "5px", "margin-bottom": "0px"},
                    scrollable=True),
                    html.Div([dbc.Row([dbc.Col(html.Div([
                        dbc.Row([dbc.Col(html.Div([dbc.Button(
                        children=html.Span([html.I(className="fa-solid fa-solid fa-solid fa-file-export",
                        style={"display": "inline-block"}),html.Div("Inputs/Downloads")],
                        style={"margin-top": "-5px", "margin-bottom": "10px"}),
                                   id="inputs-offcanvas-button",
                                   color=None, n_clicks=0,
                                   style={"margin-top": "10px"}),]), width=2),
                                 dbc.Col([html.H5("Channel selection", style={"margin-top": "12.5px"}),
                                          dcc.Dropdown(id='image_layers', multi=True,
                                     style={"margin-top": "10px"})],
                                         width=6, style={"display": "inline-block"}),
                        dbc.Col([html.H6("Canvas size"),
                                           dcc.Slider(50, 150, 5, value=100,
                                                      id='annotation-canvas-size',
                                   marks={50: 'small', 100: 'default',
                                          150: 'large'})],width=4, style={"display": "inline-block",
                                        "margin-top": "15px"})]),
                        dbc.Row([dbc.Col(width=2),
                        dbc.Col([html.Div([], style={"margin-top": "15px", "width": "100%",
                                                        "float": "left", "display": "inline-block"})], width=6)],
                                style={"display": "flex"}),
                        dbc.Row([dbc.Col([html.Div([dbc.Button(
                            children=html.Span([html.Div("Fullscreen"),
                                html.I(className="fa-solid fa-display", style={"display": "inline-block"}),
                                  ], style={"width": "100vw", "margin-top": "-5px", "margin-bottom": "10px"}),
                                             id="make-canvas-fullscreen",
                                            color=None, n_clicks=0,
                                style={"margin-left": "10px", "margin-top": "0px", "height": "100%"}),
                                dbc.Button(children=html.Span([html.Div("Auto-fit"),
                                html.I(className="fa-solid fa-arrows-left-right-to-line", style={"display": "inline-block"}),
                                  ], style={"width": "100vw", "margin-top": "-5px", "margin-bottom": "10px"}),
                                           id="autosize-canvas", color=None, n_clicks=0, style={"margin-left": "10px",
                                    "height": "100%", "width": "auto"})])],
                                         width=3, style={"display": "inline-block"}),
                                 dbc.Col(html.Div(style={"margin-top": "20px", "height": "100%",
                                                  "width": "auto"},
                                          id="bound-shower"), width=5, style={"float": "left", "display": "flex",
                                                                              }),
                                 dbc.Col([dcc.Input(id="set-x-auto-bound", type="number", value=None,
                                            placeholder="Set x-coord",
                                            style={"margin-left": "10px",
                                                   "margin-top": "15px", "width": "30%"}
                                            ),
                                  dcc.Input(id="set-y-auto-bound", type="number", value=None,
                                            placeholder="Set y-coord",
                                            style={"margin-left": "10px",
                                                   "margin-top": "20px", "width": "30%", "margin-right": "-10px"}
                                            ),
                                  dbc.Button(html.Span([html.Div("Set", style={"margin-right": "5px", "height": "150%",
                                                                               "font-size": "16px"}),
                                                        html.I(className="fa-solid fa-location-dot",
                                                                  style={"display": "inline-block",
                                                                         "margin-right": "7.5px", "margin-top": "3px"}),
                                                           ],
                                                          style={"display": "flex"}), id="activate-coord",
                                             color=None, n_clicks=0,
                                             style={"margin-left": "-75px",
                                                    "float": "right", "margin-top": "15px", "margin-right": "15px"})],
                                         width=4, style={"float": "right", "margin": "0px", "margin-top": "-5px"})
                                 ], style={"display": "flex", "margin": "0px"}),
                        html.Div([render_default_annotation_canvas(input_id="annotation_canvas")],
                                 style={"display": "flex", "justifyContent": "center"},
                                 id="canvas-div-holder"),
                    # html.Div(id='blend-color-legend', style={'whiteSpace': 'pre-line'}),
                    ]),width=9),
                        dbc.Col([
                            html.Div(
                                [html.H5("Channel Modification",
                                         style={'width': '75%', 'display': 'inline-block', "margin-top": "5px",
                                                "margin-left": "10px"}),
                                 html.Abbr("\u2753", title="Select a channel in the current blend to \nchange colour, "
                                                           "pixel intensity, or apply a filter.",
                                           style={'width': '5%', 'display': 'inline-block'}),
                                 dcc.Dropdown(id='images_in_blend', multi=False),
                                 html.Br(),
                                 daq.ColorPicker(id="annotation-color-picker", label="Current channel color",
                                                 value=dict(hex="#00ABFC", rgb=None)),
                                 dmc.Center([dmc.ColorPicker(swatches=["#FF0000", "#00FF00", "#0000FF", "#00FAFF",
                                                                     "#FF00FF", "#FFFF00", "#FFFFFF"],
                                                swatchesPerRow=7, size = 'xs', withPicker=False, id="swatch-color-picker",
                                                           fullWidth=False)]),
                                 dbc.Button(children=html.Span([html.I(className="fa-solid fa-signal",
                                style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                                html.Div("Show/hide pixel histogram")],
                                style={"display": "flex", "margin-top": "10px"}), id="show-pixel-hist",
                                className="mx-auto", color="light", n_clicks=0,
                                style={"display": "flex", "width": "auto", "align-items": "center",
                                "float": "center", "justify-content": "center"}),
                                 html.Div(dbc.Collapse(html.Div([html.H6("Pixel histogram", style={'width': '75%'}),
                                html.Div([dcc.Loading(dcc.Graph(id="pixel-hist", figure={'layout': dict(
                                xaxis_showgrid=False, yaxis_showgrid=False, xaxis=XAxis(showticklabels=False),
                                yaxis=YAxis(showticklabels=False), margin=dict(l=5, r=5, b=15, t=20, pad=0))},
                                style={'width': '60vh', 'height': '30vh', 'margin-left': '-30px'},
                                # config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                                # keep zoom and pan bars to be able to modify the histogram view
                                # 'modeBarButtonsToRemove': ['zoom', 'pan']
                                # },
                                ), type="default", fullscreen=False)])]),
                                id="pixel-hist-collapse", is_open=False), style={"minHeight": "100px"}),
                                 html.Div([dcc.RangeSlider(0, 100, 1, value=[None, None],
                                marks=dict([(i, str(i)) for i in range(0, 100, 25)]),
                                id='pixel-intensity-slider', tooltip={"placement": "top", "always_visible": True})],
                                          style={"width": "96.5%", "margin-left": "27px", "margin-top": "-50px"}),
                                 html.Br(),
                                 html.Div([dcc.Checklist(options=[' apply/refresh filter'], value=[], id="bool-apply-filter",
                                               style={"width": "85%"}),
                                 dcc.Dropdown(['median', 'gaussian'], 'median', id='filter-type',
                                              style={"width": "85%", "display": "inline-block"}),
                                 dcc.Input(id="kernel-val-filter", type="number", value=3, style={"width": "50%"})],
                                          style={"display": "inline-block", "margin": "20px"}),
                                 html.Br(),
                                 dbc.Button(
                                     children=html.Span([html.I(className="fa-solid fa-gears",
                                                                style={"display": "inline-block",
                                                                       "margin-right": "3px"}),
                                                         html.Div("Advanced canvas options"),
                                                         ]),
                                     id="blend-offcanvas-button", className="mx-auto",
                                     color=None, n_clicks=0, style={"display": "flex", "width": "auto",
                                                                    "align-items": "center",
                                                   "float": "center", "justify-content": "center"}),
                                 html.Br(),
                                 dbc.Offcanvas(id="blend-config-offcanvas", title="Advanced settings: canvas & region",
                                               placement="end", style={"width": "30%"}, backdrop=False, scrollable=True,
                                               is_open=False, children=[
                                dbc.Tabs(id='config-tabs',
                                children=[dbc.Tab(label="Configuration", tab_id='blend-config-tab',
                                children=[
                                html.Br(),
                                html.Div([daq.ToggleSwitch(label='Toggle legend', id='toggle-canvas-legend',
                                labelPosition='bottom', value=True, color="blue", style={"width": "75%",
                                "margin-left": "-15px"}),
                                daq.ToggleSwitch(label='Toggle scalebar', id='toggle-canvas-scalebar',
                                    labelPosition = 'bottom', value = True, color = "blue",
                                    style = {"width": "75%", "margin-left": "-15px"}),
                                    html.Div([html.H6("Set scalebar value", style={'width': '100%'}),
                                dcc.Input(id="custom-scale-val", type="number", value=None,
                                    style={"width": "60%", "margin-left": "30px"})],
                                    style={"display": "block"})], style={"display": "flex"}),
                                    html.Br(),
                                    html.Abbr(dcc.Checklist(options=[' show channel intensities on hover'],
                                    value=[], id="channel-intensity-hover"),
                                    title="WARNING: speed is significantly compromised with this feature, "
                                            "particularly for large images."),
                                    html.Br(),
                                    html.H6("Adjust legend/scale size"),
                                    dcc.Slider(10, 24, 1, value=16,
                                              id='legend-size-slider',
                                              marks={10: 'small', 16: 'default', 24: 'large'}),
                                    html.Br(),
                                    html.Br(),
                                    html.H5("Mask configuration"),
                                    html.Br(),
                                    html.H6("Set mask array and opacity"),
                                    dcc.Loading(
                                        dcc.Dropdown(id='mask-options', multi=False,
                                                     options=[],
                                                     style={'width': '100%',
                                                            'display': 'inline-block',
                                                            'margin-right': '-50'}),
                                        type="default", fullscreen=False),
                                    dcc.Slider(0, 100, 2.5, value=35,
                                               id='mask-blending-slider',
                                               marks={0: '0%', 25: '25%',
                                                      50: '50%', 75: '75%',
                                                      100: '100%'}),
                                    html.Div([html.Div([daq.ToggleSwitch(
                                        label='Apply mask', id='apply-mask',
                                        labelPosition='bottom', color="blue",
                                        style={"margin-left": "60px"}),
                                        html.Abbr(dcc.Checklist(
                                            options=[
                                                ' add boundary'],
                                            value=[' add boundary'],
                                            id="add-mask-boundary",
                                            style={
                                                "margin-left": "35px",
                                                "margin-top": "10px"}),
                                            title="Use this feature only if the cell "
                                                  "boundary was not derived on import"),
                                        dcc.Checklist(
                                            options=[
                                                ' show mask ID on hover'],
                                            value=[],
                                            id="add-cell-id-mask-hover",
                                            style={
                                                "margin-left": "35px",
                                                "margin-top": "10px"}),
                                    ],
                                        style={"display": "flex"})])
                                ]),
                                dbc.Tab(label="Region/Presets",
                                tab_id='region-config-tab',
                                children=[dbc.Button(children=html.Span([html.I(className="fa-solid fa-chart-area",
                                style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                                html.Div("Show/hide region statistics")], style={"display": "flex"}),
                                id="compute-region-statistics", className="mx-auto", color=None, n_clicks=0,
                                style={"margin-top": "10px"}),
                                html.Br(),
                                html.Div(dbc.Collapse(html.Div([html.H6("Selection information",
                                style={'width': '75%'}),
                                html.Div([dash_table.DataTable(id='selected-area-table',
                                columns=[{'id': p, 'name': p} for p in ['Channel', 'Mean', 'Max', 'Min']],
                                data=None)], style={"width": "85%"})]),
                                id="area-stats-collapse", is_open=False), style={"minHeight": "100px",
                                                                                 "margin-bottom": "0px"}),
                                dbc.Button(children=html.Span([html.I(className="fa-solid fa-layer-group",
                                style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                                html.Div("Add region annotation")], style={"display": "flex"}),
                                    id="region-annotation", className="mx-auto", color=None, n_clicks=0,
                                    disabled=True, style={"margin-top": "10px"}),
                                #TODO: update the logic for the button that can clear annotation shapes
                                html.Div([dbc.Button(children=html.Span([html.I(className="fa-solid fa-delete-left",
                                style={"display": "inline-block","margin-right": "7.5px","margin-top": "3px"}),
                                html.Div("Clear annotation shapes")],style={"display": "flex"}),
                                id="clear-region-annotation-shapes", className="mx-auto", color=None, n_clicks=0,
                                disabled=False, style={"margin-top": "10px"}),
                                dbc.Button(children=html.Span([html.I(className="fa-solid fa-delete-left",
                                style={"display": "inline-block", "margin-right": "7.5px",
                                "margin-top": "3px"}), html.Div("Clear ROI annotations")],
                                style={"display": "flex"}), id="clear-annotation_dict",
                                className="mx-auto", color=None, n_clicks=0, style={"margin-top": "10px", "width": "80%"})],
                                style={"display": "flex", "width": "50%"}),
                                dbc.Button(children=html.Span([html.I(className="fa-solid fa-rectangle-list",
                                style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                                html.Div("Show ROI annotations")], style={"display": "flex"}),
                                id="show-annotation-table", className="mx-auto", color=None, n_clicks=0,
                                style={"margin-top": "10px"}),
                                dbc.Modal(children=dbc.ModalBody(
                                [dash_table.DataTable(id='annotation-table', columns=[], data=None,
                                editable=False, filter_action='native')]), id="annotation-preview", size='xl'),
                                html.Br(),
                                html.Br(),
                                          dbc.Button("Create preset", id="preset-button", className="me-1"),
                                          html.Br(),
                                          dbc.Popover(dcc.Input(id="set-preset", type="text",
                                                                placeholder="Create a preset from the current channel",
                                                                value=None,
                                                                style={"width": "100%"}), target="preset-button",
                                                      trigger="hover"),
                                          html.Br(),
                                          dcc.Dropdown(options=[], value=None, id='preset-options'),
                                          dbc.Tooltip(children="", target="preset-options",
                                                      id="hover-preset-information", trigger="hover"),
                                html.Br(),
                                html.Br(),
                                dbc.Button(children=html.Span([html.I(className="fa-solid fa-download",
                                                                                style={"display": "inline-block",
                                                                                       "margin-right": "7.5px",
                                                                                       "margin-top": "3px"}),
                                                                         html.Div("Download cell annotations")],
                                                                        style={"display": "flex"}),
                                                     id="btn-download-annotations", className="mx-auto", color=None,
                                                     n_clicks=0,
                                                     style={"margin-top": "10px"}),
                                dcc.Download(id="download-edited-annotations"),
                                dbc.Button(children=html.Span([html.I(className="fa-solid fa-download",
                                style={"display": "inline-block","margin-right": "7.5px","margin-top": "3px"}),
                                html.Div("Download annotations report (PDF)")], style={"display": "flex"}),
                                id="btn-download-annot-pdf", className="mx-auto", color=None, n_clicks=0,
                                style={"margin-top": "10px"}),
                                dcc.Download(id="download-annotation-pdf"),
                                dbc.Modal(children=dbc.ModalBody(
                                [dbc.Row([dbc.Col([html.H6("Create a region annotation")], width=8),
                                          dbc.Col([html.H6("Annotate with cell type")], width=4)]),
                                 dbc.Row([dbc.Col([html.Div([dcc.Input(id="new-annotation-col", type="text",
                                value="", placeholder="Create annotation column",
                                style={"width": "50%", "margin-right": "10px", "height": "50%"}),
                                dbc.Button("Add new annotation column", id="add-annotation-col",
                                className="me-1", style={"margin-top": "-10px"})],
                                                            style={"display": "flex"})], width=8),
                                dbc.Col([dcc.Dropdown(id='quant-annotation-col',
                                multi=False, options=['ccramic_cell_annotation'],
                                    value="ccramic_cell_annotation")], width=4)]),
                                html.Br(),
                                dbc.Row([dbc.Col([html.Div([dcc.Input(id="region-annotation-name", type="text",
                                value="", placeholder="Annotation title",
                                style={"width": "65%", "margin-right": "10px", "height": "50%"}),
                                dcc.Input(id="region-annotation-body", type="text",
                                value="", placeholder="Annotation description",
                                          style={"width": "65%", "margin-right": "10px", "height": "50%"})],
                                style={"display": "flex"})], width=8),
                                dbc.Col([
                                # dcc.Dropdown(id='region-annotation-cell-types',
                                # multi=False, options=[], placeholder="Select a cell type")
                                dcc.Input(id="region-annotation-cell-types", type="text",
                                value="", placeholder="New cell type", style={"width": "65%", "margin-right": "10px",
                                    "height": "100%"})
                                ], width=4)]),
                                dbc.Row(dbc.Col(html.Div([], style={"display": "flex"}))),
                                dbc.Button("Create annotation", id="create-annotation",
                                className="me-1", style={"margin-top": "10px"})]),
                                id="region-annotation-modal", size='xl', style={"margin-left": "10px",
                                "margin-top": "15px"}),
                                html.Br(),
                                ], style={"padding": "5px"})
                                    ]),
                                     ]),
                                 html.Div([dbc.Button(
                                     children=html.Span([html.I(className="fa-solid fa-list-check",
                                                                style={"display": "inline-block",
                                                                       "margin-right": "3px"}),
                                                         html.Div("Set blend order"),
                                                         ]),
                                     id="set-sort",
                                     color=None, n_clicks=0, className="mx-auto",
                                     style={"display": "flex", "width": "auto", "align-items": "center",
                                                   "float": "center", "justify-content": "center"}),
                                     dag.AgGrid(
                                     id='blend-options-ag-grid',
                                     rowData=[],
                                     columnDefs=[{'field': 'Channel', 'rowDrag': True}],
                                     defaultColDef={"sortable": False, "filter": False},
                                     columnSize="sizeToFit", style={"height": None},
                                     dashGridOptions={
                                         "rowDragManaged": True,
                                         "animateRows": True,
                                         # "rowDragMultiRow": True,
                                         # "rowSelection": "multiple",
                                         # "rowDragEntireRow": True,
                                         "pagination": False,
                                         "domLayout": "autoHeight",
                                         # "getRowId": False
                                     })], style={"width": "90%", "margin-left": "12.5px", "height": "auto"}),
                                 html.Br(),
                                 html.Br()
                                 ],
                                style={"background-color": "#f8f9fa", "padding": "10px", "display": "inline-block",
                                       "float": "right", "width": "100%"},

                            ),

                        ],width=3),

                    ])])]),

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
                        dbc.Row(id="image-gallery-row")], style={"margin-top": "15px"}),
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
                            width=3)])])]),
                          dbc.Tab(label="Quantification/Clustering",
                                  children=[
                                html.Div([dbc.Row([
                                dbc.Col(html.Div([html.Br(),
                                                  html.H6("Cell-Level Marker Expression"),
                                                  dcc.RadioItems(['max', 'mean', 'min'], 'mean',
                                                                 inline=True, id="quantification-bar-mode"),
                                                  dcc.Graph(id="quantification-bar-full",
                                                            figure={'layout': dict(xaxis_showgrid=False,
                                                                                   yaxis_showgrid=False,
                                                                                   xaxis=XAxis(
                                                                                       showticklabels=False),
                                                                                   yaxis=YAxis(
                                                                                       showticklabels=False),
                                                                                   margin=dict(l=5, r=5, b=15,
                                                                                               t=20, pad=0)),
                                                                    })]), width=6),
                                    dbc.Col(html.Div([html.Br(),
                                    html.H6("Dimension Reduction"),
                                    html.Div([dcc.Loading(dcc.Dropdown(id='umap-projection-options', multi=False,
                                    options=[], style={"width": "175%"}), type="default", fullscreen=False),
                                    dbc.Button(children=html.Span([html.I(className="fa-solid fa-table-list",
                                    style={"display": "inline-block", "margin-right": "7.5px", "margin-top": "3px"}),
                                    html.Div("Show distribution")], style={"display": "flex"}),
                                    id="show-quant-dist", className="mx-auto", color=None, n_clicks=0)],
                                    style={"display": "flex", "width": "135%"}),
                                    dbc.Modal(children=dbc.ModalBody([dash_table.DataTable(id='quant-dist-table',
                                    columns=[], data=None, editable=False, filter_action='native')]),
                                    id="show-quant-dist-table", size='l'),
                                    dcc.Graph(id="umap-plot", figure={'layout': dict(xaxis_showgrid=False,
                                    yaxis_showgrid=False, xaxis=XAxis(showticklabels=False),
                                    yaxis=YAxis(showticklabels=False), margin=dict(l=5, r=5, b=15,t=20, pad=0)),
                                    })]), width=6)
                                      ])]),
                        dbc.Modal(children=dbc.ModalBody([html.H6("Select the cell type annotation column"),
                        dcc.Dropdown(id='cell-type-col-designation',
                            multi=False, options=[], style={'width': '100%'})]),
                        id="quantification-config-modal", size='l', style={"margin-left": "10px",
                            "margin-top": "15px"})],
                                  )])
                          ])], id='tab-annotation'),
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
        dcc.Store(id="channel-order"),
    ], style={"margin-left": "20px", "margin-right": "25px", "margin-top": "10px"}, className="dash-bootstrap")

    dash_app.enable_dev_tools(debug=True)

    init_pixel_level_callbacks(dash_app, tmpdirname, authentic_id)
    init_cell_level_callbacks(dash_app, tmpdirname, authentic_id)

    return dash_app.server
