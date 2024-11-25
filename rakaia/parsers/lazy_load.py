import os
from functools import partial
from typing import Union
from readimc import MCDFile
from rakaia.utils.pixel import split_string_at_pattern
from rakaia.parsers.spatial import check_spot_grid_multi_channel, spatial_canvas_dimensions


class SingleMarkerLazyLoaderExtensions:
    extensions = ['.h5ad', '.mcd']


def parse_files_for_lazy_loading(uploads: Union[list, dict], data_selection: str, delimiter: str= "+++"):

    """
    Parse the current upload list for extensions that support single marker lazy loading
    """
    uploads = uploads['uploads'] if isinstance(uploads, dict) and 'uploads' in uploads else uploads
    exp, slide, acq =  split_string_at_pattern(data_selection, delimiter)
    for upload in uploads:
        if any(upload.endswith(ext) for ext in SingleMarkerLazyLoaderExtensions.extensions) and exp in upload:
            return upload
    return None

class SingleMarkerLazyLoader:
    """
    Provides an option to load a single channel/marker from a supported file upload.
    """
    MATCHES = {".mcd": "mcd", ".h5ad": "h5ad"}

    def __init__(self, image_dict: dict, data_selection: str,
                session_uploads: dict,
                channels_selected: Union[str, list],
                spot_size: Union[int,float]=55,
                delimiter: str="+++"):
        self.uploads = session_uploads
        self.image_dict = image_dict
        self.delimiter = delimiter
        self.exp, self.slide, self.acq = None, None, None
        self.width, self.height, self.x_min, self.y_min = None, None, 0, 0
        if data_selection and delimiter and session_uploads:
            self.data_selection = data_selection
            self.exp, self.slide, self.acq = split_string_at_pattern(data_selection, delimiter)
            self.channel_selection = [channels_selected] if isinstance(channels_selected, str) else (
                channels_selected)
            self.spot_size = spot_size
            self.h5ad = partial(self.parse_h5ad)
            self.mcd = partial(self.parse_mcd)
            file_to_parse = parse_files_for_lazy_loading(self.uploads, self.data_selection, self.delimiter)
            if file_to_parse:
                filename, file_extension = os.path.splitext(file_to_parse)
                getattr(self, self.MATCHES[file_extension])(file_to_parse)

    def parse_h5ad(self, h5ad_filepath):
        self.width, self.height, self.x_min, self.y_min = spatial_canvas_dimensions(h5ad_filepath)
        if self.channel_selection:
            self.image_dict = check_spot_grid_multi_channel(self.image_dict, self.data_selection,
                                            h5ad_filepath, self.channel_selection, self.spot_size)

    def parse_mcd(self, mcd_filepath):
        with (MCDFile(mcd_filepath) as mcd_file):
            for slide in mcd_file.slides:
                for acq in slide.acquisitions:
                    pattern = f"{str(acq.description)}_{str(acq.id)}"
                    if pattern == self.acq:
                        self.width = int(acq.metadata['MaxX']) if 'MaxX' in acq.metadata else None
                        self.height = int(acq.metadata['MaxY']) if 'MaxY' in acq.metadata else None
                        if self.channel_selection:
                            img = mcd_file.read_acquisition(acq, strict=False)
                            for selection in self.channel_selection:
                                if not self.image_dict[self.data_selection] or \
                                        self.image_dict[self.data_selection][selection] is None:
                                    self.image_dict[self.data_selection][selection] = \
                                    img[acq.channel_names.index(selection)]

    def get_image_dict(self):
        return self.image_dict

    def get_region_dim(self):
        return self.width, self.height, self.x_min, self.y_min
