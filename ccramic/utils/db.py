import datetime

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
                                            global_filter_type, global_filter_val, global_filter_sigma):
    """
    Format a mongoDB document from a session config that can be posted to the `blend_config` collection
    in the `ccramic-db` mongoDB database
    """
    return {"user": user,
        "name": config_name,
        "channels": blend_dict,
        "config": {"blend": selected_channel_list, "filter": {"global_apply_filter": global_apply_filter,
                                                                  "global_filter_type": global_filter_type,
                                                                  "global_filter_val": global_filter_val,
                                                              "global_filter_sigma": global_filter_sigma}}}
