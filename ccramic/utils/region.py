import math
from ccramic.utils.cell_level_utils import get_min_max_values_from_zoom_box, get_min_max_values_from_rect_box
from ccramic.utils.pixel_level_utils import get_area_statistics_from_rect, get_area_statistics_from_closed_path, \
    get_bounding_box_for_svgpath, RectangularKeys
from pydantic import BaseModel
from typing import Union
import ast

class ChannelRegion:
    """
    This abstract class defines a region for a particular channel
    """
    def __init__(self, channel_array, coordinates):
        self.channel_array = channel_array
        self.coordinate_dict = coordinates
        self.mean_exp = None
        self.max_exp = None
        self.min_exp = None
        # integrated has the sum of the signal in a region
        self.integrated = None

    def compute_pixel_mean(self):
        """
        Compute the mean pixel intensity of the channel region
        """
        return self.mean_exp

    def compute_pixel_min(self):
        """
        Compute the min pixel intensity of the channel region
        """
        return self.min_exp


    def compute_pixel_max(self):
        """
        Compute the max pixel intensity of the channel region
        """
        return self.max_exp

    def compute_integrated_signal(self):
        return self.integrated

class RectangleRegion(ChannelRegion):
    """
    This class defines a channel region created using the zoom feature
    """
    def __init__(self, channel_array, coordinates, reg_type="zoom", redrawn=False):
        super().__init__(channel_array, coordinates)
        self.type = reg_type
        self.redrawn = redrawn
        self.key_dict = RectangularKeys().keys
        if self.type == "rect" and self.redrawn:
            self.required_keys = self.key_dict["rect_redrawn"]
        else:
            self.required_keys = self.key_dict[self.type]
        if all([elem in self.coordinate_dict] for elem in self.required_keys):
            try:
                if not all([elem >= 0 for elem in self.coordinate_dict.keys() if isinstance(elem, float)]):
                    raise AssertionError
                x_range_low = min(math.ceil(int(self.coordinate_dict[self.required_keys[0]])),
                                  math.ceil(int(self.coordinate_dict[self.required_keys[1]])))
                x_range_high = max(math.ceil(int(self.coordinate_dict[self.required_keys[0]])),
                                  math.ceil(int(self.coordinate_dict[self.required_keys[1]])))
                y_range_low = min(math.ceil(int(self.coordinate_dict[self.required_keys[2]])),
                                  math.ceil(int(self.coordinate_dict[self.required_keys[3]])))
                y_range_high = max(math.ceil(int(self.coordinate_dict[self.required_keys[2]])),
                                  math.ceil(int(self.coordinate_dict[self.required_keys[3]])))
                if not x_range_high >= x_range_low: raise AssertionError
                if not y_range_high >= y_range_low: raise AssertionError

                self.mean_exp, self.max_exp, self.min_exp, self.integrated = get_area_statistics_from_rect(
                    self.channel_array, x_range_low, x_range_high, y_range_low, y_range_high)
            except KeyError:
                self.mean_exp, self.max_exp, self.min_exp, self.integrated = 0, 0, 0, 0

class FreeFormRegion(ChannelRegion):
    """
    This class defines a channel region created by drawing a freeform SVG path shape
    """

    def __init__(self, channel_array, coordinates):
        super().__init__(channel_array, coordinates)
        if 'path' in self.coordinate_dict:
            self.path = self.coordinate_dict['path']
        # elif isinstance(self.coordinate_dict, dict) and \
        #         all(['shapes' in elem and 'path' in elem for elem in self.coordinate_dict.keys()]):
        #     self.path = list(self.coordinate_dict.values())[0]
        elif isinstance(self.coordinate_dict, str):
            self.path = self.coordinate_dict
        else:
            self.path = None
        if self.path is not None:
            self.mean_exp, self.max_exp, self.min_exp, self.integrated = get_area_statistics_from_closed_path(
                self.channel_array, self.path)
        else:
            self.mean_exp, self.max_exp, self.min_exp, self.integrated = 0, 0, 0, 0

class RegionAnnotation(BaseModel):
    id: str = None
    title: str = None
    body: str = None
    cell_type: str = None
    imported: bool = False
    annotation_column: str = 'ccramic_cell_annotation'
    type: str = None
    channels: list = []
    use_mask: Union[bool, list, str] = None
    mask_selection: str = None
    mask_blending_level: float = 35.0
    add_mask_boundary: Union[bool, list, str] = True

def check_for_valid_annotation_hash(annotations_dict: dict=None, roi_selection: str=None):
    """
    Check the current annotation hash table, and create a new one if id doesn't exist,
    or add in the current ROI if not present
    """
    if annotations_dict is None or len(annotations_dict) < 1:
        annotations_dict = {}
    if roi_selection and roi_selection not in annotations_dict.keys():
        annotations_dict[roi_selection] = {}
    return annotations_dict

class AnnotationPreviewGenerator:
    """
    Generates a text-based preview of an annotation to be compatible with the annotation preview table,
    which summarizes all of the current annotations in the selected ROI
    Different annotation types will have different string previews:
        - region: lists the bounding box coordinates
        - point: lists the xy coordinates
        - gating: lists the number of objects in the gate
    """

    def generate_annotation_preview(self, annot_key, annot_type="zoom"):
        """
        Generates a list of previews for each annotation in the current ROI
        """
        if annot_type == "point":
            return self.generate_point_preview(annot_key)
        elif annot_type == "gate":
            # if use gating, simply add the number of cells
            return f"{len(annot_key)} cells"
        elif annot_type in ['zoom', 'rect', 'path']:\
            return self.generate_region_preview(annot_key, annot_type)
        return None

    @staticmethod
    def generate_region_preview(key, reg_type="zoom"):
        """
        Generate a preview of a region key. Has the following general tuple structures:
        Zoom:
        (('xaxis.range[0]', 826), ('xaxis.range[1]', 836), ('yaxis.range[0]', 12), ('yaxis.range[1]', 21))
        Rectangle:
        (('x0', 826), ('x1', 836), ('y0', 12), ('y1', 21))
        svg path:
        'M670.7797603577856,478.9708311618908L675.5333177884905,487.2270098573258L676.0336922548805,'
        '481.4727034938408L668.2778880258355,479.9715800946708L668.5280752590305,479.9715800946708Z'
        """
        x_min, x_max, y_min, y_max = None, None, None, None
        if reg_type == "zoom":
            x_min, x_max, y_min, y_max = get_min_max_values_from_zoom_box(dict(key))
        elif reg_type =="rect":
            x_min, x_max, y_min, y_max = get_min_max_values_from_rect_box(dict(key))
        elif reg_type == "path":
            x_min, x_max, y_min, y_max = get_bounding_box_for_svgpath(key)
        return f"x: [{round(x_min)}, {round(x_max)}]\n y: [{round(y_min)}, {round(y_max)}]"

    @staticmethod
    def generate_point_preview(key):
        """
        Generate a preview of the click point key. Has the following general string structure:
        "{'points': [{'x': 582, 'y': 465}]}"
        """
        eval_points = ast.literal_eval(key)
        return f"x: {eval_points['points'][0]['x']}, y: {eval_points['points'][0]['y']}"
