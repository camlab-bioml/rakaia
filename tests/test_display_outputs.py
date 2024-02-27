import os.path
from ccramic.io.display import (
    RegionSummary,
    output_current_canvas_as_tiff,
    output_current_canvas_as_html,
    FullScreenCanvas,
    generate_preset_options_preview_text)
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objs as go
import plotly.express as px

def test_generate_channel_statistics_dataframe():
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.full((1000, 1000), 100),
                                                   "Nuclear": np.full((1000, 1000), 200),
                                                   "Cytoplasm": np.full((1000, 1000), 300)}}
    graph_layout = {'xaxis.range[1]': 650, 'xaxis.range[0]': 540,
                                                  'yaxis.range[1]': 800, 'yaxis.range[0]': 900}

    layers = ["DNA", "Cytoplasm"]
    aliases = {"DNA": "DNA", "Cytoplasm": "Cytoplasm", "Nuclear": "Nuclear"}

    # First stats option: when zoom is used for two channels

    stats_1 = pd.DataFrame(RegionSummary(graph_layout, upload_dict, layers, "experiment0+++slide0+++acq0",
                                                 aliases).get_summary_frame())
    assert len(stats_1) == 2
    assert list(stats_1['Max'] == [100, 300])
    assert list(stats_1['Min'] == [100, 300])
    assert list(stats_1['Mean'] == [100, 300])
    assert list(stats_1['Total'] == [1100000.0, 3300000.0])


    # Second Option: when svg path is used for one channel

    graph_layout_2 = {'shapes': [{'line': {'color': 'white', 'width': 2},
                                  'type': 'line', 'x0': 0.05, 'x1': 0.125, 'xref': 'paper', 'y0': 0.05,
                                  'y1': 0.05, 'yref': 'paper'},
                                 {'editable': True, 'label': {'text': ''},
                                  'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1,
                                  'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd',
                                  'type': 'path',
                                  'path': 'M264.9191616766467,210.12874251497007L274.7994011976048,193.06287425149702L306.2365269461078,158.032934131736'
                                          '54L299.9491017964072,145.45808383233532L276.59580838323353,114.91916167664671L265.8173652694611,'
                                          '104.14071856287426L258.6317365269461,105.937125748503L254.14071856287427,113.12275449101796L246.'
                                          '0568862275449,114.91916167664671L238.87125748502996,111.32634730538922L228.99101796407186,'
                                          '99.6497005988024L223.60179640718565,86.17664670658682L206.53592814371257,'
                                          '59.23053892215569L194.85928143712576,56.53592814371258L186.7754491017964,'
                                          '57.43413173652695L158.03293413173654,76.29640718562874L149.05089820359282,'
                                          '98.75149700598803L146.3562874251497,117.61377245508982L144.55988023952096,'
                                          '160.72754491017963L139.17065868263472,186.7754491017964L139.17065868263472,'
                                          '233.48203592814372L144.55988023952096,257.73353293413174L154.44011976047904,'
                                          '279.29041916167665L162.5239520958084,284.6796407185629L202.9431137724551,'
                                          '283.7814371257485L219.1107784431138,277.4940119760479Z'}]}

    stats_2 = pd.DataFrame(
        RegionSummary(graph_layout_2, upload_dict, ["Nuclear"], "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats_2) == 1
    assert list(stats_2['Min']) == [200]

    # Option 3: when two rectangles are drawn for two channels

    graph_layout_3 = {'shapes': [{'line': {'color': 'white', 'width': 2},
                                  'type': 'line', 'x0': 0.05, 'x1': 0.125, 'xref': 'paper',
                                  'y0': 0.05, 'y1': 0.05, 'yref': 'paper'},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4,
                                'dash': 'solid'}, 'fillcolor': 'rgba(0,0,0,0)',
                                'fillrule': 'evenodd', 'type': 'rect', 'x0': 127.4940119760479,
                                  'y0': 127.04491017964072, 'x1': 290.06886227544913, 'y1': 232.13473053892216},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1,
                                  'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type':
                                'rect', 'x0': 397.8532934131737, 'y0': 262.6736526946108,
                                  'x1': 520.0089820359282, 'y1': 443.2125748502994}]}

    stats_3 = pd.DataFrame(
        RegionSummary(graph_layout_3, upload_dict, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats_3) == 4
    assert list(stats_3['Max'] == [100, 300, 100, 300])
    assert list(stats_3['Min'] == [100, 300, 100, 300])
    assert list(stats_3['Mean'] == [100, 300, 100, 300])
    assert list(stats_3['Total']) == [1711500.0, 5134500.0, 2226300.0, 6678900.0]


    # Option 4: when an existing svg path is edited

    graph_layout_4 = {'shapes[1].path': 'M349.3502994011976,346.2065868263473L366.4161676646707,'
                                        '266.26646706586826L267.6137724550898,'
                                        '275.248502994012L257.73353293413174,277.04491017964074L234.3802395209581,'
                                        '308.4820359281437L210.12874251497004,327.3443113772455L186.7754491017964,'
                                        '336.3263473053892L184.0808383233533,339.9191616766467L185.87724550898204,'
                                        '372.2544910179641L190.3682634730539,389.32035928143716L214.6197604790419,'
                                        '428.84131736526945L244.26047904191617,454.88922155688624L287.374251497006,'
                                        '480.93712574850304L290.9670658682635,480.93712574850304L294.55988023952096,'
                                        '465.6676646706587L329.5898203592814,395.60778443113776L338.57185628742513,'
                                        '392.0149700598802L343.062874251497,392.0149700598802Z'}

    stats_4 = pd.DataFrame(
        RegionSummary(graph_layout_4, upload_dict, ["Nuclear"], "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats_4) == 1
    assert list(stats_4['Min']) == [200]


    # Option 5: when an existing rectangle is updated

    graph_layout_5 = {'shapes[1].x0': 253.2425149700599, 'shapes[1].x1': 350.248502994012,
                      'shapes[1].y0': 111.32634730538922, 'shapes[1].y1': 311.62574850299404}

    stats_5 = pd.DataFrame(
        RegionSummary(graph_layout_5, upload_dict, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())

    assert len(stats_5) == 2
    assert list(stats_5['Max'] == [100, 300])
    assert list(stats_5['Min'] == [100, 300])
    assert list(stats_5['Mean'] == [100, 300])


    # Option 6: when none of the above are called, return an empty frame

    empty_layout = {'display': None}

    stats_6 = pd.DataFrame(
        RegionSummary(empty_layout, upload_dict, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats_6) == 0


def test_generate_channel_statistics_dataframe_errors():
    """
    test when errors occur with the region parsing
    """
    upload_dict = {"experiment0+++slide0+++acq0": {"DNA": np.full((1000, 1000), 100),
                                                   "Nuclear": np.full((1000, 1000), 200),
                                                   "Cytoplasm": np.full((1000, 1000), 300)}}

    layers = ["DNA", "Cytoplasm"]
    aliases = {"DNA": "DNA", "Cytoplasm": "Cytoplasm", "Nuclear": "Nuclear"}

    # If there is a key error because the shape dimensions are malformed

    graph_layout_bad = {'shapes': [{'line': {'color': 'white', 'width': 2},
                                  'type': 'line', 'x0': 0.05, 'x1': 0.125, 'xref': 'paper',
                                  'y0': 0.05, 'y1': 0.05, 'yref': 'paper'},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4,
                                                                           'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)',
                                  'fillrule': 'evenodd', 'type': 'rect', 'x0': 127.4940119760479,
                                  'y0': 127.04491017964072, 'x1': 0, 'y1': 232.13473053892216},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1,
                                  'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type':
                                      'rect', 'x0': 397.8532934131737, 'y0': 262.6736526946108,
                                  'x1': 520.0089820359282, 'y1': 443.2125748502994}]}

    stats = pd.DataFrame(
        RegionSummary(graph_layout_bad, upload_dict, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats) == 2
    assert list(stats['Max'] == [100, 300])
    assert list(stats['Min'] == [100, 300])
    assert list(stats['Mean'] == [100, 300])


    # test when the arrays are None
    upload_dict_none = {"experiment0+++slide0+++acq0": {"DNA": None,
                                                   "Nuclear": None,
                                                   "Cytoplasm": None}}
    graph_layout = {'xaxis.range[1]': 650, 'xaxis.range[0]': 540,
                    'yaxis.range[1]': 800, 'yaxis.range[0]': 900}

    layers = ["DNA", "Cytoplasm"]
    aliases = {"DNA": "DNA", "Cytoplasm": "Cytoplasm", "Nuclear": "Nuclear"}

    stats_none = pd.DataFrame(
        RegionSummary(graph_layout, upload_dict_none, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())
    assert len(stats_none) == 0


    shape_edited = {'shapes[1].x0': 253.2425149700599, 'shapes[1].x1': 350.248502994012,
                      'shapes[1].y0': 111.32634730538922, 'shapes[1].y1': 311.62574850299404}

    stats_none_2 = pd.DataFrame(
        RegionSummary(shape_edited, upload_dict_none, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())

    assert len(stats_none_2 ) == 0

    path_edited = {'shapes[1].path': 'M349.3502994011976,346.2065868263473L366.4161676646707,'
                                        '266.26646706586826L267.6137724550898,'
                                        '275.248502994012L257.73353293413174,277.04491017964074L234.3802395209581,'
                                        '308.4820359281437L210.12874251497004,327.3443113772455L186.7754491017964,'
                                        '336.3263473053892L184.0808383233533,339.9191616766467L185.87724550898204,'
                                        '372.2544910179641L190.3682634730539,389.32035928143716L214.6197604790419,'
                                        '428.84131736526945L244.26047904191617,454.88922155688624L287.374251497006,'
                                        '480.93712574850304L290.9670658682635,480.93712574850304L294.55988023952096,'
                                        '465.6676646706587L329.5898203592814,395.60778443113776L338.57185628742513,'
                                        '392.0149700598802L343.062874251497,392.0149700598802Z'}

    stats_none_3 = pd.DataFrame(
        RegionSummary(path_edited, upload_dict_none, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())

    assert len(stats_none_3) == 0

    graph_layout_shapes = {'shapes': [{'line': {'color': 'white', 'width': 2},
                                  'type': 'line', 'x0': 0.05, 'x1': 0.125, 'xref': 'paper',
                                  'y0': 0.05, 'y1': 0.05, 'yref': 'paper'},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1, 'line': {'color': 'white', 'width': 4,
                                                                           'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)',
                                  'fillrule': 'evenodd', 'type': 'rect', 'x0': 127.4940119760479,
                                  'y0': 127.04491017964072, 'x1': 290.06886227544913, 'y1': 232.13473053892216},
                                 {'editable': True, 'label': {'text': ''}, 'xref': 'x', 'yref': 'y',
                                  'layer': 'above', 'opacity': 1,
                                  'line': {'color': 'white', 'width': 4, 'dash': 'solid'},
                                  'fillcolor': 'rgba(0,0,0,0)', 'fillrule': 'evenodd', 'type':
                                      'rect', 'x0': 397.8532934131737, 'y0': 262.6736526946108,
                                  'x1': 520.0089820359282, 'y1': 443.2125748502994}]}

    stats_none_4 = pd.DataFrame(
        RegionSummary(graph_layout_shapes, upload_dict_none, layers, "experiment0+++slide0+++acq0",
                                           aliases).get_summary_frame())

    assert len(stats_none_4) == 0


def test_output_canvas_tiff_to_file():
    canvas_image = np.full((100, 100, 3), 7)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "canvas.tiff")
        assert not os.path.exists(file_path)
        canvas_link = output_current_canvas_as_tiff(canvas_image, tmpdirname)
        assert os.path.exists(file_path)
        assert str(file_path) == str(canvas_link)
        if os.access(canvas_link, os.W_OK):
            os.remove(canvas_link)
        assert not os.path.exists(canvas_link)

def test_output_canvas_html_to_file():
    canvas_image = np.full((100, 100, 3), 7)
    fig = go.Figure(px.imshow(canvas_image)).to_dict()
    style = {"height": 100, "width": 100}
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "canvas.html")
        assert not os.path.exists(file_path)
        canvas_link = output_current_canvas_as_html(fig, style, tmpdirname)
        assert os.path.exists(file_path)
        assert str(file_path) == str(canvas_link)
        if os.access(canvas_link, os.W_OK):
            os.remove(canvas_link)
        assert not os.path.exists(canvas_link)

def test_fullscreen_canvas():
    all_white = np.full((600, 600, 3), 255).astype(np.uint8)
    canvas = go.Figure(px.imshow(all_white))
    canvas.add_shape(type="rect")
    canvas.add_annotation(x=4, y=4,
            text="This is a label",
            showarrow=False,
            yshift=10)
    assert len(canvas['layout']['shapes']) == len(canvas['layout']['annotations']) == 1
    fullscreen = FullScreenCanvas(canvas.to_dict(), {"autosize": True})
    fullscreen_canvas = fullscreen.get_canvas()
    assert len(fullscreen_canvas['layout']['shapes']) == len(fullscreen_canvas['layout']['annotations']) == 0

def test_output_preset_text():
    presets = {"preset_1": {"x_lower_bound": 1, "x_upper_bound": 10, 'filter_type': 'gaussian',
                            'filter_val': 1, 'filter_sigma': 1.0}}
    preset_preview = generate_preset_options_preview_text(presets)
    assert preset_preview == 'preset_1: \r\n l_bound: 1, y_bound: 10, ' \
                             'filter type: gaussian, filter val: 1, filter_sigma: 1.0 \r\n'
    preset_malformed = {"preset_1": {"fake_keys": None}}
    assert generate_preset_options_preview_text(preset_malformed) == ''
