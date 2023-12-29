import dash
from dash_extensions.enrich import Output, Input, State, html, Serverside
from ccramic.db.connection import AtlasDatabaseConnection
from dash.exceptions import PreventUpdate

def init_db_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with the mongoDB database
    """
    dash_app.config.suppress_callback_exceptions = True

    @dash_app.callback(Output('database-connection', 'data'),
                       #TODO: insert outputs for alert
                       Output('db-connect-alert', 'children'),
                       Output('db-connect-alert', 'is_open'),
                       Output('db-connect-alert', 'color'),
                       Input('db-connect', 'n_clicks'),
                       State('db-username', 'value'),
                       State('db-password', 'value'),
                       prevent_initial_call=True)
    def create_database_connection(connect, username, password):
        if None not in (username, password) and connect > 0:
            connection = AtlasDatabaseConnection(username=username, password=password)
            connected, ping = connection.ping_connection()
            pair = connection.username_password_pair() if connected else dash.no_update
            del connection
            return pair, ping, True, "success" if connected else "danger"
        else:
            raise PreventUpdate

    @dash_app.callback(Output('db-saved-configs', 'data'),
                       Output('db-config-options', 'options'),
                       Input('database-connection', 'data'),
                       prevent_initial_call=True)
    def populate_blend_config_list_from_database(credentials):
        """
        Import a list of blend config hash tables by user
        """
        if credentials:
            connection = AtlasDatabaseConnection(username=credentials['username'], password=credentials['password'])
            connected, ping = connection.ping_connection()
            if connected:
                configs, list = connection.blend_configs_by_user()
                return Serverside(configs), list
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate
