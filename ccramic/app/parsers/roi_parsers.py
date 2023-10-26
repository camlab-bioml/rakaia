import pandas as pd
import h5py
from pathlib import Path
from tifffile import TiffFile
import os
from ..utils.pixel_level_utils import *
from readimc import MCDFile, TXTFile
import random
import numpy as np

def generate_multi_roi_images_from_query(dataset_selection, session_config, blend_dict,
                                         currently_selected_channels, num_queries=5, rois_exclude=None,
                                         predefined_indices=None):
    """
    Generate a gallery of images for multiple ROIs using the current parameters of the current ROI
    Important: ignores the current ROI
    """
    if rois_exclude is None:
        rois_exclude = []
    try:
        roi_images = {}
        # get the index of the file from the experiment number in the event that there are multiple uploads
        # file_path = None
        # for files_uploaded in session_config['uploads']:
        #     if str(Path(files_uploaded).stem) == basename:
        #         file_path = files_uploaded
        files = [file for file in session_config['uploads'] if str(file).endswith('.mcd')]
        random.shuffle(files)
        if files is not None and len(files) > 0:
            queries_obtained = 0
            for file_path in files:
                basename = str(Path(file_path).stem)
                with MCDFile(file_path) as mcd_file:
                    # generate a random number of roi indices to query
                    # query_length = min(len(dataset_options), num_queries)
                    # queries = random.sample(range(0, len(dataset_options)), query_length)
                    slide_index = 0
                    for slide_inside in mcd_file.slides:
                        if predefined_indices is not None:
                            query_selection = predefined_indices
                        else:
                            if num_queries > len(slide_inside.acquisitions):
                                query_selection = range(0, len(slide_inside.acquisitions))
                            else:
                                query_selection = random.sample(range(0, len(slide_inside.acquisitions)), num_queries)
                        for query in query_selection:
                            acq = slide_inside.acquisitions[query]
                            if f"{basename}+++slide{slide_index}+++{acq.description}" not in rois_exclude:
                                channel_names = acq.channel_names
                                channel_index = 0
                                img = mcd_file.read_acquisition(acq)
                                acq_image = []
                                for channel in img:
                                    # if the channel is in the current blend, use it
                                    if channel_names[channel_index] in currently_selected_channels and \
                                        channel_names[channel_index] in blend_dict.keys():
                                        with_preset = apply_preset_to_array(channel,
                                                        blend_dict[channel_names[channel_index]])
                                        recoloured = np.array(recolour_greyscale(with_preset,
                                                                    blend_dict[channel_names[channel_index]][
                                                                        'color'])).astype(np.float32)
                                        acq_image.append(recoloured)
                                    channel_index += 1
                                summed_image = sum([image for image in acq_image]).astype(np.float32)
                                summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)
                                label = f"{basename}+++slide{slide_index}+++{acq.description}"
                                roi_images[label] = summed_image
                                queries_obtained += 1
                            else:
                                num_queries += 1
                            if len(roi_images) == num_queries:
                                break
                        else:
                            slide_index += 1
                            continue
                        break
                    else:
                        continue
                    break
            return roi_images
        else:
            return None
    # return roi_images
    except (KeyError, AssertionError):
        return None
