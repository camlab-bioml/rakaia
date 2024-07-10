
import tempfile
import os
from rakaia.io.pdf import AnnotationPDFWriter
import numpy as np
from PIL import Image

def test_output_annotations_pdf():
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
        output_pdf = AnnotationPDFWriter(annotations_dict, layers_dict, data_selection,
                    mask_config=None, aliases=aliases, dest_dir=tmpdirname).generate_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)

        mask_config = {"mask": {"array": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB')),
                                "raw": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB')),
                                "boundary": np.array(Image.fromarray(np.zeros((500, 500))).convert('RGB'))}}
        rect_tuple = tuple(sorted({'x0': 50, 'x1': 100,
                                    'y0': 50, 'y1': 100}.items()))
        annotations_dict = {data_selection: {rect_tuple: {'title': 'Title', 'body': 'body',
                                                           'cell_type': 'cell_type', 'imported': False,
                                                           'type': 'rect', 'channels': ['channel_1'],
                                                           'use_mask': True,
                                                           'mask_selection': "mask",
                                                           'mask_blending_level': 35,
                                                           'add_mask_boundary': True}}}

        output_pdf = AnnotationPDFWriter(annotations_dict, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases, dest_dir=tmpdirname).generate_annotation_pdf()

        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(file_path)

        # test on an svgpath, make the image and masks bigger than 500x500
        layers_dict = {"exp1+++slide0+++roi_1":
                           {"channel_1": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB'))}}
        mask_config = {"mask": {"array": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB')),
                                "raw": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB')),
                                "boundary": np.array(Image.fromarray(np.zeros((1000, 1000))).convert('RGB'))}}
        svgpath = 'M670.7797603577856,478.9708311618908L675.5333177884905,487.2270098573258L676.0336922548805,' \
                  '492.2307545212258L671.2801348241755,500.73712044985575L669.7790114250056,' \
                  '501.98805661583077L668.0277007926405,501.4876821494408L665.7760156938856,' \
                  '499.2359970506858L663.5243305951306,497.9850608847108L662.2733944291556,' \
                  '496.23375025234577L661.7730199627656,492.9813162208108L661.7730199627656,' \
                  '491.2300055884458L662.7737688955456,490.47944388886077L665.0254539943006,' \
                  '490.47944388886077L665.7760156938856,486.4764481577408L665.2756412274956,' \
                  '484.72513752537577L664.7752667611055,482.7236396598158L666.0262029270806,' \
                  '477.2195205295258L667.2771390930556,480.7221417942558L667.5273263262505,' \
                  '481.4727034938408L668.2778880258355,479.9715800946708L668.5280752590305,479.9715800946708Z'

        annotations_dict = {data_selection: {svgpath: {'title': 'Title', 'body': 'body',
                                                          'cell_type': 'cell_type', 'imported': False,
                                                          'type': 'path', 'channels': ['channel_1'],
                                                          'use_mask': True,
                                                          'mask_selection': "mask",
                                                          'mask_blending_level': 35,
                                                          'add_mask_boundary': True}}}

        output_pdf = AnnotationPDFWriter(annotations_dict, layers_dict, data_selection,
                    mask_config=None, aliases=aliases, dest_dir=tmpdirname).generate_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)

        blend_dict = {"channel_1": {'color': '#FFFFFF'}}

        output_pdf = AnnotationPDFWriter(annotations_dict, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases, dest_dir=tmpdirname,
                    blend_dict=blend_dict).generate_annotation_pdf()
        assert os.path.exists(output_pdf)
        if os.access(output_pdf, os.W_OK):
            os.remove(output_pdf)

        assert not os.path.exists(output_pdf)
        assert AnnotationPDFWriter({"exp1+++slide0+++roi_1": {}}, layers_dict, data_selection,
                    mask_config=mask_config, aliases=aliases, dest_dir=tmpdirname,
                    blend_dict=blend_dict).generate_annotation_pdf() is None
        assert not os.path.exists(output_pdf)
