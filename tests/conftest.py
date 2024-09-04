import pytest
import os
from functools import wraps
import tempfile
import numpy as np
from rakaia.entrypoint import init_app

@pytest.fixture(scope="module")
def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))

@pytest.fixture(scope="module")
def rakaia_flask_test_app():
    app = init_app(cli_config={'use_local_dialog': False, 'use_loading': True,
                               'persistence': True, 'swatches': None, 'array_store_type': 'float',
                               'serverside_overwrite': False, 'is_dev_mode': False, 'cache_dest': tempfile.gettempdir(),
                               'threads': 8})
    app.config.update({
        "TESTING": True,
    })
    yield app


@pytest.fixture(scope="module")
def client(rakaia_flask_test_app):
    return rakaia_flask_test_app.test_client()

def skip_on(exception, reason="Default reason"):
    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(reason)

        return wrapper

    return decorator_func


@pytest.fixture(scope="module")
def annotation_hash():
    return {'Patient1+++slide0+++pos1_1': {(('xaxis.range[0]', 384.3802395209581),
                                                        ('xaxis.range[1]', 487.6736526946108),
                                                        ('yaxis.range[0]', 426.1467065868263),
                                                        ('yaxis.range[1]', 322.8532934131736)):
                                                           {'title': 'test',
                                                            'id': 'annot_1',
                                                            'body': 'test',
                                                            'cell_type': 'cell type 1',
                                                            'imported': False,
                                                            'annotation_column': 'rakaia_cell_annotation',
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
                                                           'imported': False, 'id': 'annot_2',
                                                           'annotation_column': 'rakaia_cell_annotation',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']}, (
                                                           ('x0', 198.45209580838323), ('x1', 440.9670658682635),
                                                           ('y0', 40.81736526946108), ('y1', 155.7874251497006)): {
            'title': 'test', 'body': 'test', 'cell_type': 'cell type 3', 'imported': False, 'id': 'annot_3',
            'annotation_column': 'broad', 'type': 'rect', 'channels': ['Ho165'], 'use_mask': None,
            'mask_selection': None, 'mask_blending_level': 35, 'add_mask_boundary': [' add boundary']},
                                                       'M97.85329341317365,422.55389221556885L114.02095808383234,431.53592814371257L136.47604790419163,456.685628742515L164.32035928143713,500.69760479041923L168.811377245509,514.1706586826348L167.9131736526946,533.9311377245509L159.82934131736528,541.116766467066L127.4940119760479,542.9131736526947L113.12275449101796,538.4221556886229L90.66766467065868,524.0508982035929L61.026946107784426,500.69760479041923L40.368263473053894,470.1586826347306L34.97904191616767,453.0928143712575L34.97904191616767,434.2305389221557L53.84131736526947,423.45209580838326L54.73952095808384,423.45209580838326Z': {
                                                           'title': 'test', 'body': 'test', 'cell_type': 'cell type 4',
                                                           'imported': False, 'annotation_column': 'broad',
                                                           'id': 'annot_4',
                                                           'type': 'path', 'channels': ['Ho165'], 'use_mask': None,
                                                           'mask_selection': None, 'mask_blending_level': 35,
                                                           'add_mask_boundary': [' add boundary']},
                                                       "{'points': [{'curveNumber': 0, 'x': 235, 'y': 124, 'color': "
                                                       "{'0': 0, '1': 0, '2': 255, '3': 1}, 'colormodel': 'rgba256', "
                                                       "'z': {'0': 0, '1': 0, '2': 255, '3': 1}, 'bbox': "
                                                       "{'x0': 503.63, 'x1': 504.75, 'y0': 448.61, 'y1': 448.61}}]}":
                                                           {'title': None, 'body': None, 'cell_type': 'immune',
                                                            'imported': False, 'id': 'annot_5',
                                                            'annotation_column': 'rakaia_cell_annotation',
                                                            'type': 'point', 'channels': None,
                                                            'use_mask': None, 'mask_selection': None,
                                                            'mask_blending_level': None, 'add_mask_boundary': None},
                                                       (101, 102, 103, 104, 105):
                                                           {'title': None, 'body': None, 'cell_type': 'gated_test',
                                                            'imported': False, 'id': 'annot_6',
                                                            'annotation_column': 'rakaia_cell_annotation',
                                                            'type': 'gate', 'channels': None,
                                                            'use_mask': True, 'mask_selection': 'mask',
                                                            'mask_blending_level': None, 'add_mask_boundary': None},
                                                       'bad_annotation': 'bad_annotation_entry'
                                                       }}


@pytest.fixture(scope="module")
def annotation_hash_filtered(annotation_hash):
    annotation_hash['Patient1+++slide0+++pos1_1'] = {key: value for key, value in
                annotation_hash['Patient1+++slide0+++pos1_1'].items() if isinstance(value, dict)}
    return annotation_hash

@pytest.fixture(scope="module")
def annotation_hash_filtered_no_gate(annotation_hash_filtered):
    annotation_hash_filtered['Patient1+++slide0+++pos1_1'] = {key: value for key, value in
        annotation_hash_filtered['Patient1+++slide0+++pos1_1'].items() if not value['type'] == 'gate'}
    return annotation_hash_filtered

@pytest.fixture(scope="module")
def svgpath():
    return 'M670.7797603577856,478.9708311618908L675.5333177884905,487.2270098573258L676.0336922548805,' \
              '492.2307545212258L671.2801348241755,500.73712044985575L669.7790114250056,' \
              '501.98805661583077L668.0277007926405,501.4876821494408L665.7760156938856,' \
              '499.2359970506858L663.5243305951306,497.9850608847108L662.2733944291556,' \
              '496.23375025234577L661.7730199627656,492.9813162208108L661.7730199627656,' \
              '491.2300055884458L662.7737688955456,490.47944388886077L665.0254539943006,' \
              '490.47944388886077L665.7760156938856,486.4764481577408L665.2756412274956,' \
              '484.72513752537577L664.7752667611055,482.7236396598158L666.0262029270806,' \
              '477.2195205295258L667.2771390930556,480.7221417942558L667.5273263262505,' \
              '481.4727034938408L668.2778880258355,479.9715800946708L668.5280752590305,479.9715800946708Z'

@pytest.fixture(scope="module")
def channel_hash():
    return {"experiment0+++slide0+++acq0": {"DNA": np.zeros((600, 600, 3)),
                                                       "Nuclear": np.zeros((600, 600, 3)),
                                                       "Cytoplasm": np.zeros((600, 600, 3))},
                                              "experiment0+++slide0+++acq1": {"DNA": np.zeros((600, 600, 3)),
                                                       "Nuclear": np.zeros((600, 600, 3)),
                                                       "Cytoplasm": np.zeros((600, 600, 3))}}
@pytest.fixture(scope="module")
def channel_hash_2():
    return {"experiment0+++slide0+++acq0": {"DNA": np.array([0, 0, 0, 0]),
                                                   "Nuclear": np.array([1, 1, 1, 1]),
                                                   "Cytoplasm": np.array([2, 2, 2, 2]),
                                                   "Other_Nuclear": np.array([3, 3, 3, 3])},
                   "experiment0+++slide0+++acq1": {"DNA": np.array([3, 3, 3, 3]),
                                                   "Nuclear": np.array([4, 4, 4, 4]),
                                                   "Cytoplasm": np.array([5, 5, 5, 5]),
                                                   "Other_Nuclear": np.array([6, 6, 6, 6])}}

@pytest.fixture(scope="module")
def recursive_aliases():
    aliases = {}
    for i in range(5):
        aliases[f"channel_{i}"] = f"initial_label_{i}"
    return aliases

@pytest.fixture(scope="module")
def recursive_aliases_2():
    aliases = {}
    for i in range(5):
        aliases[f"channel_{i}"] = f"rakaia_label_{i}"
    return aliases
@pytest.fixture(scope="module")
def recursive_gallery_children(recursive_aliases):
    children = []
    for key, value in recursive_aliases.items():
        children.append({"props": {"children": [{"children": value, "className": "card-text", 'id': key},
                                    {"children": f'Add {value} to canvas', 'target': {'index': key}}]}})
    return children
