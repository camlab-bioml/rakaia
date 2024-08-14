import dash
import pandas as pd
import matplotlib
import dash_uploader as du
from dash_extensions.enrich import Output, Input, State
from dash.exceptions import PreventUpdate
from dash import ctx

from rakaia.inputs.metadata import metadata_association_plot
from rakaia.io.readers import DashUploaderFileReader
from rakaia.io.session import SessionServerside


def init_metadata_level_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with metadata/patient level analysis
    (metadata variable association)

    :param dash_app: the dash proxy server wrapped in the parent Flask app
    :param tmpdirname: the path for the tmpdir for tmp storage for the session
    :param authentic_id: uuid string identifying the particular app invocation
    :param app_config: Dictionary of session options passed through CLI
    :return: None
    """
    dash_app.config.suppress_callback_exceptions = True
    matplotlib.use('agg')
    OVERWRITE = app_config['serverside_overwrite']

    @du.callback(Output('custom-metadata', 'data'),
                 id='upload-custom-metadata')
    def get_umap_upload_from_drag_and_drop(status: du.UploadStatus):
        files = DashUploaderFileReader(status).return_filenames()
        if files: return SessionServerside(pd.read_csv(files[0]).to_dict(orient="records"), key="custom_metadata", use_unique_key=OVERWRITE)
        raise PreventUpdate

    @dash_app.callback(Output('meta-x-axis', 'options'),
                       Output('meta-y-axis', 'options'),
                       Output('meta-grouping', 'options'),
                       Input('custom-metadata', 'data'),
                       prevent_initial_call=True)
    def populate_metadata_association_dropdowns(custom_metadata):
        if custom_metadata:
            cols = list(pd.DataFrame(custom_metadata).columns)
            return cols, cols, cols

    @dash_app.callback(Input('swap-association-axes', 'n_clicks'),
                       State('meta-x-axis', 'value'),
                       State('meta-y-axis', 'value'),
                       Output('meta-x-axis', 'value'),
                       Output('meta-y-axis', 'value'),
                       prevent_initial_call=True)
    def invert_association_axes(invert, x_axis, y_axis):
        return y_axis if y_axis else dash.no_update, x_axis if x_axis else dash.no_update


    @dash_app.callback(Output('metadata-association-plot', 'figure'),
                       State('custom-metadata', 'data'),
                       Input('meta-x-axis', 'value'),
                       Input('meta-y-axis', 'value'),
                       Input('meta-grouping', 'value'),
                       prevent_initial_call=True)
    def generate_metadata_association_plot(custom_metadata, x_axis, y_axis, grouping):
        if custom_metadata and x_axis and y_axis:
            return metadata_association_plot(custom_metadata, x_axis, y_axis, grouping, ctx.triggered_id == "meta-grouping")
        raise PreventUpdate
