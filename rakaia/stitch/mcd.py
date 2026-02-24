"""
Module for parsing MCD files to retrieve relevant stitching information such as
global slide coordinates systems, slide parameters, ROI coordinates, etc.
"""
from typing import Union
from pathlib import Path
from readimc import MCDFile
from readimc.data.slide import Slide, Acquisition
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
        self._slide_match, self._acq_match = self._get_current_roi_mcd_slide(self._acq)

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

    def _get_current_roi_mcd_slide(self, roi_identifier: Union[str, None]=None) -> Union[
                                    tuple[Slide, Acquisition], tuple[None, None]]:
        """
        Get the MCD file slide for the current ROI, or None if it doesn't exist

        :param roi_identifier: ROI Identifier to match. From MCD, should be formatted as `acq.description_acq.id`

        :return: Tuple[`Slide`, `Acquisition`] if the ROI identifier matches to a slide, or None otherwise
        """
        if self._roi_from_mcd and self._path:
            with MCDFile(self._path) as mcd_open:
                for slide in mcd_open.slides:
                    for acq in slide.acquisitions:
                        if str(roi_identifier) == f"{acq.description}_{acq.id}":
                            return slide, acq
        return None, None

    def get_roi_slide_params(self):
        """
        Get the width and height for the MCD file slide backing the current ROI. IMPORTANT: this will trim
        the width and height to the min and max coordinates of the ROIs found on the same slide in the same MCD file.

        :return: Tuple (width, height) in pixels for the slide
        """
        if self._slide_match is not None and isinstance(self._slide_match, Slide):
            x_coords = []
            y_coords = []
            for acq in self._slide_match.acquisitions:
                for point in acq.roi_points_um:
                    x_coords.append(int(point[0]))
                    y_coords.append(int(point[1]))
            return int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
        return None, None

    def _get_mcd_slide_min_max_coords(self, type_coord: str= "max"):
        """
        Get the min or max width and height in pixels for the current slide, without any offsets. Requires iterating
        over all the acquisitions on the slide to find the min anx max coordinates covered

        :param type_coord: String representing the min or max coordinates to be returned

        :return: Tuple (x, y) coordinate for the slide, either the min or the max.
        """
        if self._slide_match is not None and isinstance(self._slide_match, Slide):
            x_coords = []
            y_coords = []
            for acq in self._slide_match.acquisitions:
                for point in acq.roi_points_um:
                    x_coords.append(int(point[0]))
                    y_coords.append(int(point[1]))
            return (int(max(x_coords)), int(max(y_coords))) if type_coord == "max" else \
                    (int(min(x_coords)), int(min(y_coords)))
        return None, None

    @staticmethod
    def invert_mcd_coord(coord: Union[int, float],
                           coord_max: Union[int, float],
                           pixel_size: Union[int, float]=1.0):
        """
        Invert an MCD coordinate to be compatible with numpy indexing. Currently, applies only to the y-axis.

        :param coord: Individual coordinate to invert
        :param coord_max: Maximum coordinate in the coordinate space for inversion
        :param pixel_size: Number of microns per pixel

        :return: Inverted coordinate as an integer
        """
        return int(int(coord_max - coord) / int(pixel_size))

    def get_roi_coord_min(self):
        """
        Get the min x and y position of the current ROI relative to the underlying slide. These
        coordinates can be used to set the spot for the stitching

        :Return: (min-x, min-y) positions for the current ROI in the corresponding slide, to set the
        starting position for stitching.
        """
        if self._slide_match and self._acq_match:
            points_translated_x = []
            points_translated_y = []
            pixel_size = int(min(self._acq_match.pixel_size_x_um, self._acq_match.pixel_size_y_um))
            max_width, max_height = self._get_mcd_slide_min_max_coords()
            min_x, min_y = self._get_mcd_slide_min_max_coords("min")
            for point in self._acq_match.roi_points_um:
                points_translated_x.append(int(point[0]))
                points_translated_y.append(self.invert_mcd_coord(int(point[1]), max_height, pixel_size))
            return int(min(points_translated_x) - min_x), min(points_translated_y)
        return None, None
