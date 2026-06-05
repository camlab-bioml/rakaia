"""
Module with functions for parsing and updating stitched images from the dataset gallery tiles
"""
from typing import Union
import numpy as np
from rakaia.stitch import cur_roi_slide_matches_stitch, update_stitch_cache_with_blend
from rakaia.stitch.cosmx import cosmx_local_fov_position, cosmx_global_slide_boundaries
from rakaia.stitch.mcd import MCDAcqCoordinateParser, roi_identifier_to_steinbock_id
from rakaia.utils.pixel import split_string_at_pattern
from rakaia.utils.session import roi_from_anndata_file

class ROIGalleryStitchParser:

    FORMATS_ACCEPTED = ['mcd', 'cosmx']
    """
    Parses the ROIs currently held in the dataset gallery and determines if they should be added
    to the current stitch image. Currently, supports ROIS from MCD or CosMX Anndata with specific `uns` key slots

    :param stitch_cache: Dictionary cache of current session stitched images (RGB)
    :param stitch_selection: String selection of the current stitch image to modify from the cache
    :param roi_images: List of dataset identifiers currently rendered in the gallery
    :param session_filepaths: Dictionary of raw imaging file uploads
    :param roi_selection: String identifier for the current ROI in the canvas
    :param delimiter: String delimiter to split the roi parameter into filename, slide, and ROI identifier
    """

    def __init__(self, stitch_cache: Union[dict, None], stitch_selection: Union[str, None],
                roi_images: Union[list, None] = None, session_filepaths: Union[dict, list, None]=None,
                roi_selection: Union[str, None]=None,
                delimiter: str = "+++"):
        self._stitch_cache = stitch_cache
        self._stitch_selection = stitch_selection
        self._roi_images = roi_images
        self._roi_selection = roi_selection
        self._session_filepaths = session_filepaths['uploads'] if 'uploads' in session_filepaths else session_filepaths
        self._delimiter = delimiter
        # set this when the one of the accepted formats is detected
        self._supported_type_in_gallery = None
        self._invert_y = False
        self._detect_accepted_format()


    def _anndata_roi_selection_to_exp_name(self, roi: Union[str, None]=None):
        """
        Get the file/experiment name from an `Anndata` ROI selection

        :param roi: String identifier for the ROI to check

        :return: String file basename if the ROI is from `Anndata`, or `None` otherwise
        """
        if roi_from_anndata_file(self._session_filepaths, roi, self._delimiter):
            exp, slide, acq = split_string_at_pattern(roi, self._delimiter)
            return str(exp)
        return None

    def _detect_accepted_format(self):
        """
        Detect and set the accepted format

        :return: None: sets the supported type class attribute based on the accepted formats
        """
        # TODO: should type check be done just on the file uploads, or what's actually in the gallery on parse?
        # Would there ever be a situation where a mix of mcd and Anndata would be allowed in the same session?
        self._supported_type_in_gallery = 'mcd' if any(str(upload).endswith('.mcd') for
                        upload in self._session_filepaths) else self._supported_type_in_gallery
        if not self._supported_type_in_gallery:
            self._supported_type_in_gallery = 'cosmx' if any(str(upload).endswith('.h5ad') for
                                                upload in self._session_filepaths) else self._supported_type_in_gallery
        # imp: for now, invert the y to match the orientation of the FOVs for stitching with cosmx
        self._invert_y = True if self._supported_type_in_gallery == 'cosmx' else False

    def get_gallery_identifiers(self):
        """
        Get the gallery identifiers, return as {'names': [list of identifiers]}

        :return: Tuple: dictionary of ROI names, None, and the current ROI selection in list format
        """
        # always set the query indices to None
        if self._supported_type_in_gallery == 'mcd':
            roi_indices = {'names': [roi_identifier_to_steinbock_id(roi, self._delimiter) for roi in self._roi_images if
                                     (roi != self._roi_selection and roi_identifier_to_steinbock_id(roi, self._delimiter))]}
            return roi_indices, None, [self._roi_selection]
        if self._supported_type_in_gallery == 'cosmx':
            roi_indices = {'names': [self._anndata_roi_selection_to_exp_name(roi) for roi in self._roi_images if
                                     (roi != self._roi_selection)]}
            return {'names': [roi for roi in roi_indices['names'] if roi is not None]}, None, [self._roi_selection]
        return None, None, None

    def update_stitch_from_gallery_thumbnails(self, roi_images: Union[dict, list, None]=None):
        """
        Add dataset gallery ROI RGB images to the currently selected stitch based on an accepted format

        :param roi_images: List of dataset identifiers currently rendered in the gallery

        :return: Updated image cache (note that the method returns the entire dictionary cache, not just the updated image)
        """
        if None not in (self._stitch_cache, self._stitch_selection, roi_images) and str(self._stitch_selection) in self._stitch_cache:
            stitch_h, stitch_w = self._stitch_cache[self._stitch_selection].shape[0], self._stitch_cache[self._stitch_selection].shape[1]
            for roi_id, roi_arr in roi_images.items():
                x_min, y_min = MCDAcqCoordinateParser(self._session_filepaths, roi_id, self._delimiter).get_roi_coord_min() if \
                    self._supported_type_in_gallery == 'mcd' else (cosmx_local_fov_position(roi_from_anndata_file(
                    self._session_filepaths, roi_id, self._delimiter)) if roi_from_anndata_file(
                    self._session_filepaths, roi_id, self._delimiter) else (None, None))
                roi_slide_w, roi_slide_h = MCDAcqCoordinateParser(self._session_filepaths, roi_id,
                                                                  self._delimiter).get_roi_slide_boundary_point() if \
                    self._supported_type_in_gallery == 'mcd' else (cosmx_global_slide_boundaries(roi_from_anndata_file(
                    self._session_filepaths, roi_id, self._delimiter)) if roi_from_anndata_file(
                    self._session_filepaths, roi_id, self._delimiter) else (None, None))
                # check here that the underlying ROI slide dimensions match the dimensions of the current stitch image?
                # i.e. if the gallery contains ROIs from multiple MCDs
                if None not in (x_min, y_min, roi_slide_w, roi_slide_h) and cur_roi_slide_matches_stitch(roi_slide_h, roi_slide_w, stitch_h, stitch_w):
                    image_add = np.flip(roi_arr.astype(np.uint8), axis=0) if self._invert_y else roi_arr.astype(np.uint8)
                    self._stitch_cache = update_stitch_cache_with_blend(self._stitch_cache, self._stitch_selection, x_min, y_min, image_add)
        return self._stitch_cache
