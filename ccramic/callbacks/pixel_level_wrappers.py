import dash

GLOBAL_FILTER_KEYS = ["global_apply_filter", "global_filter_type", "global_filter_val", "global_filter_sigma"]

def parse_global_filter_values_from_json(config_dict):
    """
    parse the global filter values from a config JSON
    """
    global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma = \
        dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if 'filter' in config_dict and all([elem in config_dict['filter'].keys() for elem in GLOBAL_FILTER_KEYS]):
        global_apply_filter = config_dict['filter']['global_apply_filter']
        global_filter_type = config_dict['filter']['global_filter_type']
        global_filter_val = config_dict['filter']['global_filter_val']
        global_filter_sigma = config_dict['filter']['global_filter_sigma']
    return global_apply_filter, global_filter_type, global_filter_val, global_filter_sigma
