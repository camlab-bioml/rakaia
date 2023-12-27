import os
import json
import h5py
import pandas as pd
def write_blend_config_to_json(dest_dir, blend_dict, blend_layer_list, global_apply_filter,
                               global_filter_type, global_filter_val, global_filter_sigma):
    """
    Write the session blend configuration dictionary to a JSON file
    """
    param_json_path = str(os.path.join(dest_dir, 'param.json'))
    with open(param_json_path, "w") as outfile:
        dict_write = {"channels": blend_dict, "config":
            {"blend": blend_layer_list, "filter":
                {"global_apply_filter": global_apply_filter, "global_filter_type": global_filter_type,
                 "global_filter_val": global_filter_val, "global_filter_sigma": global_filter_sigma}}}
        json.dump(dict_write, outfile)
    return param_json_path

def write_session_data_to_h5py(dest_dir, metadata_frame, data_dict, data_selection, blend_dict, mask=None):
    """
    Write the current data dictionary and blend configuration to an h5py file
    """
    # TODO: add the global filter and blend list to the h5py output
    relative_filename = os.path.join(dest_dir, 'data.h5')
    hf = None
    try:
        hf = h5py.File(relative_filename, 'w')
    except OSError:
        os.remove(relative_filename)
    if hf is None:
        hf = h5py.File(relative_filename, 'w')

    meta_to_write = pd.DataFrame(metadata_frame) if metadata_frame is not None else \
        pd.DataFrame(data_dict['metadata'])
    for col in meta_to_write:
        meta_to_write[col] = meta_to_write[col].astype(str)
    hf.create_dataset('metadata', data=meta_to_write.to_numpy())
    hf.create_dataset('metadata_columns', data=meta_to_write.columns.values.astype('S'))
    hf.create_group(data_selection)
    for key, value in data_dict[data_selection].items():
        if key not in hf[data_selection]:
            hf[data_selection].create_group(key)
            if 'image' not in hf[data_selection][key] and value is not None:
                # use the mask if provided
                if mask is not None:
                    value[~mask] = 0
                hf[data_selection][key].create_dataset('image', data=value)
                if blend_dict is not None and key in blend_dict.keys():
                    for blend_key, blend_val in blend_dict[key].items():
                        data_write = str(blend_val) if blend_val is not None else "None"
                        hf[data_selection][key].create_dataset(blend_key, data=data_write)
                else:
                    pass
    try:
        hf.close()
    except:
        pass

    return str(relative_filename)
