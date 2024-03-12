import pandas as pd
from ccramic.io.session import JSONSessionDocument

def match_db_config_to_request_str(db_config_list: list, db_config_selection: str):
    """
    Iterate over the list of configs documents generated from mongoDB and match the
    name key to the in-app request
    """
    for config in db_config_list:
        if config['name'] == db_config_selection:
            return config
    return None

def extract_alias_labels_from_db_document(db_blend_config, session_metadata=None, return_type="dict"):
    """
    Extract the alias labels from a mongoDB document. Labels should be held in the `alias` key
    for every channel in the `channels` slot
    """
    labels = []
    try:
        for key in db_blend_config['channels'].keys():
            if 'alias' in db_blend_config['channels'][key].keys():
                labels.append(db_blend_config['channels'][key]['alias'])
            else:
                labels.append(key)
        session_metadata = pd.DataFrame(session_metadata)
        session_metadata['ccramic Label'] = labels
        return pd.DataFrame(session_metadata).to_dict(orient='records') if \
            return_type == "dict" else pd.DataFrame(session_metadata)
    except (KeyError, TypeError, ValueError):
        pass
    return session_metadata

def preview_dataframe_from_db_config_list(config_list):
    """
    Generate a dataframe preview of the configs that are available from a list of configs dictionaries
    imported from mongoDB
    """
    preview = {"Names": [], "Panel Length": [], "Selected Channels": [], "Filter": []}
    for result in config_list:
        preview["Names"].append(result['name'])
        preview["Panel Length"].append(len(result['channels'].keys()))
        selected_channels = ""
        selected_index = 0
        for channel in result['config']['blend']:
            try:
                label = result['channels'][channel]['alias']
            except KeyError:
                label = channel
            # TODO: use the label alias instead of the internal label for better readability
            delimiter_channel = " \\\n " if selected_index < (len(result['config']['blend']) - 1) else ""
            selected_channels = selected_channels + str(label) + delimiter_channel
            selected_index += 1
        preview["Selected Channels"].append(selected_channels)
        filters = ""
        filter_index = 0
        for key, value in result['config']['filter'].items():
            delimiter_filter = " \\\n " if filter_index < (len(result['config']['filter']) - 1) else ""
            filters = filters + f"{str(key)}: {str(value)}" + delimiter_filter
            filter_index += 1
        preview["Filter"].append(filters)
    return preview

def format_blend_config_document_for_insert(user, config_name, blend_dict, selected_channel_list, global_apply_filter,
                                            global_filter_type, global_filter_val, global_filter_sigma,
                                            data_selection: str=None, cluster_assignments: dict=None,
                                            aliases: dict=None, gating_dict: dict=None):
    """
    Format a mongoDB document from a session config that can be posted to the `blend_config` collection
    in the `ccramic-db` mongoDB database
    """
    return JSONSessionDocument("db", user, config_name, blend_dict, selected_channel_list, global_apply_filter,
                                         global_filter_type, global_filter_val, global_filter_sigma,
                                         data_selection, cluster_assignments, aliases, gating_dict).get_document()
