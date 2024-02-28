import tempfile
import pytest
import platform
from ccramic.io.annotation_outputs import *

def test_output_annotations_masks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        annotations_dict = {'Patient1+++slide0+++pos1_1': {(('xaxis.range[0]', 384.3802395209581),
                                                        ('xaxis.range[1]', 487.6736526946108),
                                                        ('yaxis.range[0]', 426.1467065868263),
                                                        ('yaxis.range[1]', 322.8532934131736)):
                                                        {'title': 'test',
                                                        'body': 'test',
                                                        'cell_type': 'cell type 1',
                                                        'imported': False,
                                                        'annotation_column': 'ccramic_cell_annotation',
                                                        'type': 'zoom',
                                                        'channels': ['Ho165'],
                                                        'use_mask': None,
                                                        'mask_selection': None,
                                                        'mask_blending_level': 35,
                                                        'add_mask_boundary': [' add boundary']},
                                                       'M216.41616766467067,157.58383233532933L235.27844311377245,'
                                                       '185.42814371257487L240.6676646706587,'
                                                       '210.57784431137725L241.56586826347305,239.32035928143713L'
                                                       '241.56586826347305,254.58982035928145L233.48203592814372,'
                                                       '270.75748502994014L207.43413173652695,293.2125748502994L'
                                                       '189.47005988023952,299.5L161.625748502994,297.7035928143713L'
                                                       '143.66167664670658,290.5179640718563L129.29041916167665,'
                                                       '275.248502994012L119.41017964071857,256.3862275449102L'
                                                       '117.61377245508982,224.94910179640718L132.88323353293413,'
                                                       '188.12275449101796L143.66167664670658,186.32634730538922L'
                                                       '174.2005988023952,185.42814371257487L179.58982035928145,'
                                                       '166.56586826347305L184.0808383233533,154.88922155688624L'
                                                       '185.87724550898204,153.99101796407186Z': {
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
                                                           'add_mask_boundary': [' add boundary']},
                                                           "{'points': [{'curveNumber': 0, 'x': 235, 'y': 124, 'color': "
                                                           "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
                                                           "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
                                                           "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
                                                               {'title': None, 'body': None, 'cell_type': 'immune',
                                                                'imported': False,
                                                                'annotation_column': 'ccramic_cell_annotation',
                                                                'type': 'point', 'channels': None,
                                                                'use_mask': None, 'mask_selection': None,
                                                                'mask_blending_level': None, 'add_mask_boundary': None}
                                                           }}

        assert not os.path.exists(os.path.join(tmpdirname, "annotation_masks.zip"))
        canvas_mask = np.full((600, 600), 7)
        output_dir = AnnotationMaskWriter(annotations_dict, tmpdirname,
                                                 'Patient1+++slide0+++pos1_1', (600, 600),
                                                 canvas_mask=canvas_mask).write_annotation_masks()
        assert os.path.exists(os.path.join(output_dir))


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Skip the test output of point annotations CSV in Windows (different base64 format)")
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
    point_annotations = export_point_annotations_as_csv(1, 'pos1_1', annotations_dict, 'Patient1+++slide0+++pos1_1',
                                    None, False, None, image_dict,
                                    authentic_id, tmpdirname)
    assert point_annotations == {'base64': False, 'content': 'ROI,x,y,annotation_col,annotation\n'
            'pos1_1,235,124,ccramic_cell_annotation,immune\n', 'filename': 'point_annotations.csv', 'type': None}

    mask_dict = {'mask': {'raw': image}}

    point_annotations = export_point_annotations_as_csv(1, 'pos1_1', annotations_dict, 'Patient1+++slide0+++pos1_1',
                                                        mask_dict, True, 'mask', image_dict,
                                                        authentic_id, tmpdirname)

    assert point_annotations == {'base64': False, 'content': 'ROI,x,y,annotation_col,annotation,mask_cell_id\n'
            'pos1_1,235,124,ccramic_cell_annotation,immune,0\n', 'filename': 'point_annotations.csv', 'type': None}


    point_annotations = export_point_annotations_as_csv(1, 'pos1_1', annotations_dict, 'Patient1+++slide0+++pos1_1',
                                                        mask_dict, True, 'mask', None,
                                                        authentic_id, tmpdirname)

    # assert that the mask id is not included if the image to compare cannot be found
    assert point_annotations == {'base64': False, 'content': 'ROI,x,y,annotation_col,annotation\n'
            'pos1_1,235,124,ccramic_cell_annotation,immune\n', 'filename': 'point_annotations.csv', 'type': None}

    # assert no update occurs if one of the keys is malformed

    with pytest.raises(PreventUpdate):
        export_point_annotations_as_csv(0, 'pos1_1', annotations_dict,
                                        'Patient1+++slide0+++pos1_1',
                                        mask_dict, True, 'mask', None,
                                        authentic_id, tmpdirname)

        annotations_dict_malformed = {'Patient1+++slide0+++pos1_1': {
        "{'points': [{'curveNumber': 0, 'fake_x': 235, 'y': 124, 'color': "
        "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
        "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
        "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
            {'title': None, 'body': None, 'cell_type': 'immune', 'imported': False,
             'annotation_column': 'ccramic_cell_annotation', 'type': 'point', 'channels': None,
             'use_mask': None, 'mask_selection': None, 'mask_blending_level': None, 'add_mask_boundary': None}
        }}
        export_point_annotations_as_csv(1, 'pos1_1', annotations_dict_malformed,
                                                        'Patient1+++slide0+++pos1_1',
                                                        mask_dict, True, 'mask', None,
                                                        authentic_id, tmpdirname)

        # annotations_dict_empty = {'Patient1+++slide0+++pos1_1': {}}
        # export_point_annotations_as_csv(1, 'pos1_1', annotations_dict_empty,
        #                                                 'Patient1+++slide0+++pos1_1',
        #                                                 mask_dict, True, 'mask', None,
        #                                                 authentic_id, tmpdirname)
        #
        # annotations_malformed_2 = {'Patient1+++slide0+++pos1_1': {
        #     "{'points': [{'curveNumber': 0, 'x': 235, 'y': 124, 'color': "
        #     "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
        #     "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
        #     "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
        #         {'title': None, 'body': None, 'cell_type': 'immune', 'imported': False,
        #          'annotation_column': 'ccramic_cell_annotation', 'fake_type': 'point', 'channels': None,
        #          'use_mask': None, 'mask_selection': None, 'mask_blending_level': None, 'add_mask_boundary': None}
        # }}
        #
        # export_point_annotations_as_csv(1, 'pos1_1', annotations_malformed_2,
        #                                 'Patient1+++slide0+++pos1_1',
        #                                 mask_dict, True, 'mask', None,
        #                                 authentic_id, tmpdirname)


def test_output_region_writer(get_current_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        annotations_dict = {'Patient1+++slide0+++pos1_1': {(('xaxis.range[0]', 384.3802395209581),
                                                        ('xaxis.range[1]', 487.6736526946108),
                                                        ('yaxis.range[0]', 426.1467065868263),
                                                        ('yaxis.range[1]', 322.8532934131736)):
                                                        {'title': 'test',
                                                        'body': 'test',
                                                        'cell_type': 'cell type 1',
                                                        'imported': False,
                                                        'annotation_column': 'ccramic_cell_annotation',
                                                        'type': 'zoom',
                                                        'channels': ['Ho165'],
                                                        'use_mask': None,
                                                        'mask_selection': None,
                                                        'mask_blending_level': 35,
                                                        'add_mask_boundary': [' add boundary']},
                                                       'M542.6242205274294,468.7997148823149L576.8682073859937,501.4130356999952L581.1079390922921,' \
              '501.08690249181836L588.9351360885354,497.1733039936968L596.7623330847787,494.56423832828233' \
              'L603.9372636646683,495.8687711609895L606.5463293300828,495.2165047446359L612.0905938690884,' \
              '488.6938405810999L615.3519259508564,483.8018424584478L617.634858408094,483.8018424584478L' \
              '620.5700572816853,485.7586417075087L624.1575225716301,485.43250849933185L630.0279203188126,' \
              '479.88824396032624L636.2244512741718,467.82131525778453L636.2244512741718,460.3202514697181L' \
              '635.2460516496415,450.536255224414L634.9199184414646,445.6442571017619L620.5700572816853,' \
              '439.1215929382259L612.4167270772653,430.9682627338058L610.1337946200276,424.11946536209297L' \
              '608.5031285791435,418.5752008230873L606.5463293300828,414.9877355331425L599.0452655420163,' \
              '407.8128049532528L590.8919353375962,410.42187061866724L583.3908715495297,410.0957374104904L' \
              '577.8466070105242,407.1605385368992L568.06261076522,407.1605385368992L559.9092805608,' \
              '409.11733778596L554.3650160217943,410.42187061866724L527.6220929512965,410.0957374104904L' \
              '518.490363122346,408.1389381614296L513.9244982078708,408.4650713696064L507.7279672525115,' \
              '413.68320270043523L507.7279672525115,417.9229344067337Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 2',
                                                           'imported': False,
                                                           'annotation_column': 'ccramic_cell_annotation',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': True,
                                                           'mask_selection': 'mask', 'mask_blending_level': 35,
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
                                                           'add_mask_boundary': [' add boundary']},
                                                           "{'points': [{'curveNumber': 0, 'x': 235, 'y': 124, 'color': "
                                                           "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
                                                           "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
                                                           "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
                                                               {'title': None, 'body': None, 'cell_type': 'immune',
                                                                'imported': False,
                                                                'annotation_column': 'ccramic_cell_annotation',
                                                                'type': 'point', 'channels': None,
                                                                'use_mask': None, 'mask_selection': None,
                                                                'mask_blending_level': None, 'add_mask_boundary': None}
                                                           }}

        # of the annotations present, there is only one path that has a mask to match
        mask_dict = {"mask": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}
        assert not os.path.exists(os.path.join(tmpdirname, "region_annotations.csv"))
        region_writer = AnnotationRegionWriter(annotations_dict, 'Patient1+++slide0+++pos1_1', mask_dict)
        region_writer.write_csv(dest_dir=tmpdirname)
        region_frame = pd.DataFrame(region_writer.region_object_frame)
        assert len(region_frame) == 55
        assert 142 in region_frame['cell_id'].to_list()
        assert 'cell type 2' in region_frame['annotation'].to_list()
        assert 'ccramic_cell_annotation' in region_frame['annotation_col'].to_list()

        gated_cell_tuple = (102, 154, 134, 201, 209, 244)
        # annotate using gated cell method
        annotations_dict_gate = {'Patient1+++slide0+++pos1_1': {
            gated_cell_tuple:
                {'title': 'None', 'body': 'None',
                 'cell_type': 'mature', 'imported': False, 'type': 'gate',
                 'annotation_column': 'gating_test', 'mask_selection': 'mask'}}}

        region_writer_gate = AnnotationRegionWriter(annotations_dict_gate, 'Patient1+++slide0+++pos1_1', mask_dict)
        region_writer_gate.write_csv(dest_dir=tmpdirname)
        region_frame = pd.DataFrame(region_writer_gate.region_object_frame)
        assert 'mature' in region_frame['annotation'].to_list()
        assert len(region_frame.index[region_frame['annotation'] == 'mature'].tolist()) == len(gated_cell_tuple) == \
            len(region_frame)

        empty_dict = {'Patient1+++slide0+++pos1_1': {}}
        region_writer = AnnotationRegionWriter(empty_dict, 'Patient1+++slide0+++pos1_1', mask_dict)
        region_writer.write_csv(dest_dir=tmpdirname)
        region_frame = pd.DataFrame(region_writer.region_object_frame)
        assert region_frame.empty
