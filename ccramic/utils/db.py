import pandas as pd

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
    preview = {"Names": [], "Panel": [], "Selection": [], "Filter": []}
    for result in config_list:
        preview["Names"].append(result['name'])
        preview["Panel"].append(len(result['channels'].keys()))
        selected_channels = ""
        for channel in result['config']['blend']:
            selected_channels = selected_channels + str(channel) + " \\\n "
        preview["Selection"].append(selected_channels)
        filters = ""
        for key, value in result['config']['filter'].items():
            filters = filters + f"{str(key)}: {str(value)}" + " \\\n "
        preview["Filter"].append(filters)
    return preview

def format_blend_config_document_for_insert(user, config_name, blend_dict, selected_channel_list, global_apply_filter,
                                            global_filter_type, global_filter_val, global_filter_sigma,
                                            aliases: dict=None):
    """
    Format a mongoDB document from a session config that can be posted to the `blend_config` collection
    in the `ccramic-db` mongoDB database
    """
    # TODO: add the alias to the config
    if aliases is not None:
        for key in blend_dict.keys():
            if key in aliases.keys():
                blend_dict[key]['alias'] = aliases[key]
    return {"user": user,
        "name": config_name,
        "channels": blend_dict,
        "config": {"blend": selected_channel_list, "filter": {"global_apply_filter": global_apply_filter,
                                                                  "global_filter_type": global_filter_type,
                                                                  "global_filter_val": global_filter_val,
                                                              "global_filter_sigma": global_filter_sigma}}}
