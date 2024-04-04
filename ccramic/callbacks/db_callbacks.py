import dash
from dash_extensions.enrich import Output, Input, State
from ccramic.db.connection import AtlasDatabaseConnection
from dash.exceptions import PreventUpdate
from ccramic.utils.db import preview_dataframe_from_db_config_list
import pandas as pd
from dash import ctx
from ccramic.io.session import SessionServerside

def init_db_callbacks(dash_app, tmpdirname, authentic_id, app_config):
    """
    Initialize the callbacks associated with the mongoDB database
    """
    dash_app.config.suppress_callback_exceptions = True

    @dash_app.callback(Output('database-connection', 'data'),
                       Output('db-connect-alert', 'children'),
                       Output('db-connect-alert', 'is_open'),
                       Output('db-connect-alert', 'color'),
                       Input('db-connect', 'n_clicks'),
                       State('db-username', 'value'),
                       State('db-password', 'value'),
                       State('db-connection-string', 'value'),
                       prevent_initial_call=True)
    def create_database_connection(connect, username, password, conn_string):
        if None not in (username, password, conn_string) and connect > 0 and len(conn_string) > 0:
            try:
                connection = AtlasDatabaseConnection(conn_string, username, password)
                connected, ping = connection.create_connection()
                pair = connection.username_password_pair() if connected else dash.no_update
                # connection.close()
                return pair, ping, True, "success" if connected else "danger"
            except AttributeError:
                return dash.no_update, f"Invalid database connection string", True, "danger"
        raise PreventUpdate

    @dash_app.callback(Output('db-saved-configs', 'data'),
                       Output('db-config-options', 'options'),
                       Output('db-config-preview-table', 'data'),
                       Output('db-config-preview-table', 'columns'),
                       Input('database-connection', 'data'),
                       State('db-connection-string', 'value'),
                       prevent_initial_call=True)
    def populate_blend_config_list_from_database(cred, conn_string):
        """
        Import a list of blend config hash tables by user
        """
        if cred:
            connection = AtlasDatabaseConnection(conn_string, cred['username'], cred['password'])
            connected, ping = connection.create_connection()
            if connected:
                configs, list_configs = connection.blend_configs_by_user()
                config_preview = preview_dataframe_from_db_config_list(configs)
                return SessionServerside(configs, key="dg-config-list"), \
                    list_configs, pd.DataFrame(config_preview).to_dict(orient='records'), \
                    [{'id': p, 'name': p, 'editable': False, "presentation": "markdown"} for p in config_preview.keys()]
            # raise PreventUpdate
        raise PreventUpdate

    @dash_app.callback(
        Output("db-config-preview", "is_open"),
        Input('view-db-config-list', 'n_clicks'),
        [State("db-config-preview", "is_open")])
    def toggle_db_config_preview_table(n1, is_open):
        """
        Toggle open the dataframe containing the preview of imported configs from mongoDB
        """
        if n1: return not is_open
        return is_open

    @dash_app.callback(Output('db-config-name', 'value'),
                       Input('db-config-options', 'value'),
                       prevent_initial_call=True)
    def load_selected_db_config_name_into_save(selected_db_config):
        """
        If a config file from the database is loaded, replace the name input used to save the configuration
        """
        return selected_db_config if selected_db_config is not None else None

    @dash_app.callback(Output('db-config-submit-alert', 'children'),
                       Output('db-config-submit-alert', 'is_open'),
                       Output('db-config-submit-alert', 'color'),
                       Input('db-save-cur-config', 'n_clicks'),
                       Input('db-remove-select-config', 'n_clicks'),
                       State('database-connection', 'data'),
                       State('db-config-name', 'value'),
                       State('blending_colours', 'data'),
                       State('image_layers', 'value'),
                       State('bool-apply-global-filter', 'value'),
                       State('global-filter-type', 'value'),
                       State("global-kernel-val-filter", 'value'),
                       State("global-sigma-val-filter", 'value'),
                       State('cluster-colour-assignments-dict', 'data'),
                       State('data-collection', 'value'),
                       State('alias-dict', 'data'),
                       State('gating-dict', 'data'),
                       State('db-connection-string', 'value'))
    def insert_or_remove_configuration(save_to_db, remove_from_db, cred, db_config_name, blend_dict, channel_selection,
                            global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma,
                            cluster_assignments, data_selection, aliases, gating_dict, conn_string):
        """
        Save the current session configuration (blend dictionary and parameters) as a mongoDB document to the db
        or
        remove the current config from the database if it exists
        """
        try:
            if ctx.triggered_id == "db-save-cur-config" and save_to_db and None not in (cred, db_config_name, blend_dict):
                connection = AtlasDatabaseConnection(conn_string, cred['username'], cred['password'])
                connection.create_connection()
                connection.insert_blend_config(db_config_name, blend_dict, channel_selection, global_apply_filter,
                                               global_filter_type, global_filter_val, global_filter_sigma,
                                               data_selection, cluster_assignments, aliases, gating_dict)
                return f"{db_config_name} submitted successfully", True, "success"
            elif ctx.triggered_id == "db-remove-select-config" and remove_from_db > 0 and None not in (db_config_name, cred):
                connection = AtlasDatabaseConnection(conn_string, cred['username'], cred['password'])
                connection.create_connection()
                connection.remove_blend_document_by_name(db_config_name)
                # connection.close()
                return f"{db_config_name} removed successfully", True, "success"
        except AttributeError:
            return f"Invalid database connection string. Action not performed", True, "danger"
        raise PreventUpdate
