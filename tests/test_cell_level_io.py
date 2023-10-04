import numpy as np

from ccramic.app.io.annotation_outputs import *
import os
import tempfile


def test_output_annotations_masks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        annotations_dict = {'Patient1+++slide0+++pos1_1': {(('xaxis.range[0]', 384.3802395209581),
                                                        ('xaxis.range[1]', 487.6736526946108),
                                                        ('yaxis.range[0]', 426.1467065868263),
                                                        ('yaxis.range[1]', 322.8532934131736)): {'title': 'test',
                                                                                                 'body': 'test',
                                                                                                 'cell_type': 'cell type 1',
                                                                                                 'imported': False,
                                                                                                 'annotation_column': 'ccramic_cell_annotation',
                                                                                                 'type': 'zoom',
                                                                                                 'channels': ['Ho165'],
                                                                                                 'use_mask': None,
                                                                                                 'mask_selection': None,
                                                                                                 'mask_blending_level': 35,
                                                                                                 'add_mask_boundary': [
                                                                                                     ' add boundary']},
                                                       'M216.41616766467067,157.58383233532933L235.27844311377245,185.42814371257487L240.6676646706587,210.57784431137725L241.56586826347305,239.32035928143713L241.56586826347305,254.58982035928145L233.48203592814372,270.75748502994014L207.43413173652695,293.2125748502994L189.47005988023952,299.5L161.625748502994,297.7035928143713L143.66167664670658,290.5179640718563L129.29041916167665,275.248502994012L119.41017964071857,256.3862275449102L117.61377245508982,224.94910179640718L132.88323353293413,188.12275449101796L143.66167664670658,186.32634730538922L174.2005988023952,185.42814371257487L179.58982035928145,166.56586826347305L184.0808383233533,154.88922155688624L185.87724550898204,153.99101796407186Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 2',
                                                           'imported': False,
                                                           'annotation_column': 'ccramic_cell_annotation',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']}, (
                                                       ('x0', 198.45209580838323), ('x1', 440.9670658682635),
                                                       ('y0', 40.81736526946108), ('y1', 155.7874251497006)): {
            'title': 'test', 'body': 'test', 'cell_type': 'cell type 3', 'imported': False,
            'annotation_column': 'broad', 'type': 'rect', 'channels': ['Ho165'], 'use_mask': None,
            'mask_selection': None, 'mask_blending_level': 35, 'add_mask_boundary': [' add boundary']},
                                                       'M97.85329341317365,422.55389221556885L114.02095808383234,431.53592814371257L136.47604790419163,456.685628742515L164.32035928143713,500.69760479041923L168.811377245509,514.1706586826348L167.9131736526946,533.9311377245509L159.82934131736528,541.116766467066L127.4940119760479,542.9131736526947L113.12275449101796,538.4221556886229L90.66766467065868,524.0508982035929L61.026946107784426,500.69760479041923L40.368263473053894,470.1586826347306L34.97904191616767,453.0928143712575L34.97904191616767,434.2305389221557L53.84131736526947,423.45209580838326L54.73952095808384,423.45209580838326Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 4',
                                                           'imported': False, 'annotation_column': 'broad',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']}}}

        assert not os.path.exists(os.path.join(tmpdirname, "annotation_masks.zip"))
        output_dir = export_annotations_as_masks(annotations_dict, tmpdirname,
                                                 'Patient1+++slide0+++pos1_1', (600, 600, 1))
        assert os.path.exists(os.path.join(output_dir))

def test_output_point_annotations_as_csv():
    with tempfile.TemporaryDirectory() as tmpdirname:
        annotations_dict = {'Patient1+++slide0+++pos1_1': {
    "{'points': [{'curveNumber': 0, 'x': 235, 'y': 124, 'color': "
    "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
    "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
    "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
    {'title': None, 'body': None, 'cell_type': 'immune', 'imported': False,
     'annotation_column': 'ccramic_cell_annotation', 'type': 'point', 'channels': None,
     'use_mask': None, 'mask_selection': None, 'mask_blending_level': None, 'add_mask_boundary': None}
        }}

    authentic_id = "sasfadfadfdf"
    # assert not os.path.exists(os.path.join(tmpdirname, authentic_id, 'downloads', 'annotation_masks', ''))
    image = np.zeros((600, 600))
    image_dict = {'Patient1+++slide0+++pos1_1': {'channel_1': image}}
    point_annotations = export_point_annotations_as_csv(1, annotations_dict, 'Patient1+++slide0+++pos1_1',
                                    None, False, None, image_dict,
                                    authentic_id, tmpdirname)
    assert point_annotations == {'content': 'x,y,annotation_col,annotation\n235,124,ccramic_cell_annotation,immune\n',
                                 'filename': 'point_annotations.csv', 'type': None, 'base64': False}

    mask_dict = {'mask': {'raw': image}}

    point_annotations = export_point_annotations_as_csv(1, annotations_dict, 'Patient1+++slide0+++pos1_1',
                                                        mask_dict, True, 'mask', image_dict,
                                                        authentic_id, tmpdirname)
    assert point_annotations == {'base64': False, 'content': 'x,y,annotation_col,annotation,mask\n'
            '235,124,ccramic_cell_annotation,immune,0\n', 'filename': 'point_annotations.csv', 'type': None}
