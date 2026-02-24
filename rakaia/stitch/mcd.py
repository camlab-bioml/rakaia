"""
Module for parsing MCD files to retrieve relevant stitching information such as
global slide coordinates systems, slide parameters, ROI coordinates, etc.
"""
from typing import Union
from pathlib import Path
from rakaia.utils.pixel import split_string_at_pattern

class MCDAcqCoordinateParser:
    """
    Parses for slide parameters and/or individual ROI-level global coordinates to support image stitching for a specific ROI
    derived from an MCD file.

    :param session_filepaths: List or dictionary of file uploads in the current session.
    :param roi: String identifier for the current ROI in the canvas
    :param delimiter: String delimiter to split the roi parameter into filename, slide, and ROI identifier
    """

    def __init__(self, session_filepaths: Union[dict, list, None],
                roi: Union[str, None]=None, delimiter: str="+++"):
        self._roi = roi
        self._delimiter = delimiter
        try:
            self._exp, self._slide, self._acq = split_string_at_pattern(self._roi, self._delimiter)
        except (AttributeError, ValueError):
            self._exp, self._slide, self._acq = None, None, None
        self._session_files = session_filepaths['uploads'] if (isinstance(session_filepaths, dict) and
                                            'uploads' in session_filepaths) else session_filepaths
        self._roi_from_mcd, self._path = self._is_roi_from_mcd(self._exp, self._session_files)

    @staticmethod
    def _is_roi_from_mcd(roi_filepath: Union[str, None]=None,
                        session_filepaths: Union[dict, list, None]=None):
        """
        Check if the current ROI is derived from an MCD files that contains global coordinates.
        If so, save the path

        :param roi_filepath: String identifier for the basename of file backing the current ROI
        :param session_filepaths: List or dictionary of file uploads in the current session.

        :return: Tuple containing boolean status (if the current ROI is from an MCD file), and the full filepath (if MCD).
        """
        if None not in (roi_filepath, session_filepaths):
            for session_file in session_filepaths:
                if str(Path(session_file).stem == str(roi_filepath)) and (
                        str(Path(session_file)).endswith('.mcd')):
                    return True, str(session_file)
        return False, None

    def get_mcd_status(self):
        """
        Get the status of a session ROI (if it is derived from an MCD file or not).

        :return: Tuple containing boolean status (if the `roi` parameter is from an MCD file), and the full filepath (if MCD).
        """
        return self._roi_from_mcd, self._path

    def get_mcd_slide_params_by_roi(self):
        """
        Get the slide parameters for a given ROI, such as the pixel width and height. Can be used
        for initiate a new stitched slide that can accommodate the current ROI.
        """
        if self._roi_from_mcd and self._path:
            raise NotImplementedError("Need to implement the logic to get the slide coordinates per ROI")
        return None, None

    def get_mcd_roi_global_coordinates(self):
        """
        Get the global coordinates for an acquisition/ROI from MCD as it relates to the slide that it comes from.
        Can be used to position the ROI in a new stitched image
        """
        if self._roi_from_mcd and self._path:
            raise NotImplementedError("Need to implement the logic to get the ROI global coordinates from the MCD slide.")
        return None
