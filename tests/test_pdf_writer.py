import tempfile
import os
from rakaia.io.pdf import AnnotationPDFWriter
import numpy as np
from PIL import Image

def test_output_annotations_pdf(svgpath):
    """
    test that the output of the annotations pdf produces a valid file
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "rakaia_test_annotations.pdf")
        assert not os.path.exists(file_path)
        data_selection = "exp1+++slide0+++roi_1"
        range_tuple = tuple(sorted({'xaxis.range[0]': 50, 'xaxis.range[1]': 100,
                    'yaxis.range[0]': 50, 'yaxis.range[1]': 100}.items()))
        annotations_dict = {data_selection: {range_tuple: {'title': 'Title', 'body': 'body',
                                                               'cell_type': 'cell_type', 'imported': False,
                                                            'type': 'zoom', 'channels': ['channel_1'],
                                                             'use_mask': False,
                                                             'mask_selection': None,
                                                             'mask_blending_level': 35,
                                                             'add_mask_boundary': False}}}
        layers_dict = {"exp1+++slide0+++roi_1":
                           {"channel_1": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB'))}}
        aliases = {"channel_1": "channel_1"}
        blend_dict = {"channel_2": {"color": "black"}}
        output_pdf = AnnotationPDFWriter(tmpdirname, annotations_dict, layers_dict, data_selection,
                    mask_config=None, aliases=aliases, blend_dict=blend_dict).write_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)

        mask_config = {"mask": {"raw": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB'))}}
        rect_tuple = tuple(sorted({'x0': 50, 'x1': 100,
                                    'y0': 50, 'y1': 100}.items()))
        annotations_dict = {data_selection: {rect_tuple: {'title': 'Title', 'body': 'body',
                                                           'cell_type': 'cell_type', 'imported': False,
                                                           'type': 'rect', 'channels': ['channel_1'],
                                                           'use_mask': True,
                                                           'mask_selection': "mask",
                                                           'mask_blending_level': 35,
                                                           'add_mask_boundary': True}}}

        output_pdf = AnnotationPDFWriter(tmpdirname, annotations_dict, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases).write_annotation_pdf()

        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(file_path)

        # test on an svgpath, make the image and masks bigger than 500x500
        layers_dict = {"exp1+++slide0+++roi_1":
                           {"channel_1": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB'))}}
        mask_config = {"mask": {"raw": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB'))}}
        annotations_dict = {data_selection: {svgpath: {'title': 'Title', 'body': 'body',
                                                          'cell_type': 'cell_type', 'imported': False,
                                                          'type': 'path', 'channels': ['channel_1'],
                                                          'use_mask': True,
                                                          'mask_selection': "mask",
                                                          'mask_blending_level': 35,
                                                          'add_mask_boundary': True}}}

        output_pdf = AnnotationPDFWriter(tmpdirname, annotations_dict, layers_dict, data_selection,
                    mask_config=None, aliases=aliases).write_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)

        blend_dict = {"channel_1": {'color': '#FFFFFF'}}

        output_pdf = AnnotationPDFWriter(tmpdirname, annotations_dict, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases,
                    blend_dict=blend_dict).write_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)
        assert AnnotationPDFWriter(tmpdirname, {"exp1+++slide0+++roi_1": {}}, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases,
                    blend_dict=blend_dict).write_annotation_pdf() is None
        assert not os.path.exists(output_pdf)
