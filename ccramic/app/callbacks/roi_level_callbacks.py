import dash

from .cell_level_wrappers import *
from ..parsers.roi_parsers import *
import dash_bootstrap_components as dbc
from dash import html, ALL

def init_roi_level_callbacks(dash_app, tmpdirname, authentic_id):
    """
    Initialize the callbacks associated with ROi level queries
    """

    @dash_app.callback(Output('dataset-query-gallery-row', 'children'),
                       State('image_layers', 'value'),
                       State('data-collection', 'value'),
                       State('blending_colours', 'data'),
                       State('session_config', 'data'),
                       Input('execute-dataset-query', 'n_clicks'),
                       State('data-collection', 'options'),
                       State('dataset-query-number', 'value'),
                       prevent_initial_call=True)
    def generate_roi_images_from_query(currently_selected, data_selection, blend_colour_dict,
                                                         session_config, execute_query, dataset_options, num_queries):
        if None not in (currently_selected, data_selection, blend_colour_dict,
                                                         session_config) and execute_query > 0:
            images = generate_multi_roi_images_from_query(data_selection, session_config, blend_colour_dict,
                                                    currently_selected, dataset_options, int(num_queries))
            row_children = []
            for key, value in images.items():
                # add the dimensions to the label
                label = f"{key}: {value.shape[1]}x{value.shape[0]}"
                row_children.append(dbc.Col(dbc.Card([dbc.CardBody([html.B(label, className="card-text"),
                                                                html.Br(), dbc.Button("Load in canvas",
                                                                                      id={'type': 'data-query-gallery',
                                                                                          'index': key},
                                                                                      outline=True, color="dark",
                                                                                      className="me-1", size="sm",
                                                                                      style={"margin-top": "15px"})]),
                                                  dbc.CardImg(src=Image.fromarray(value).convert('RGB'),
                                                              bottom=True)]), width=3))
            return row_children
        else:
            raise PreventUpdate

    @dash_app.callback(
        Output('data-collection', 'value', allow_duplicate=True),
        Output('pixel-level-analysis', 'active_tab', allow_duplicate=True),
        Input({'type': 'data-query-gallery', "index": ALL}, "n_clicks"),
        State('data-collection', 'options'),
        prevent_initial_call=True)
    # @cache.memoize())
    def load_roi_through_query_click(roi_query, dataset_options):
        if dataset_options is not None and not all([elem is None for elem in roi_query]):
            index_from = ctx.triggered_id["index"]
            if index_from in dataset_options:
                return index_from, "pixel-analysis"
            else:
                raise PreventUpdate
        raise PreventUpdate
