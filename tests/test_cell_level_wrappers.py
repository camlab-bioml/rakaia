import pytest

import os
import tifffile
from ccramic.callbacks.cell_level_wrappers import *

def test_basic_callback_import_annotations_quantification_frame(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    bounds = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'zoom',
                  'annotation_column': 'ccramic_cell_annotation'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = callback_add_region_annotation_to_quantification_frame(annotations_dict,
                                        measurements.to_dict(orient="records"), "roi_1", None, False, None)
    quantification_frame = pd.DataFrame(quantification_frame)

    assert list(quantification_frame["ccramic_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                        (quantification_frame["y_max"] == 20)]) == ['mature']
    assert isinstance(serverside, Serverside)

    with pytest.raises(PreventUpdate):
        callback_add_region_annotation_to_quantification_frame(None,
                                                               measurements.to_dict(orient="records"), "roi_1", None,
                                                               False, None)

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    bounds = {'x0': 826, 'x1': 836, 'y0': 12, 'y1': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'rect',
                  'annotation_column': 'ccramic_cell_annotation'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = callback_add_region_annotation_to_quantification_frame(annotations_dict,
                                                                                              measurements.to_dict(
                                                                                                  orient="records"),
                                                                                              "roi_1", None, False,
                                                                                              None)
    quantification_frame = pd.DataFrame(quantification_frame)

    assert list(quantification_frame["ccramic_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                                (quantification_frame["y_max"] == 20)]) == ['mature']

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'different', 'imported': False, 'type': 'rect',
                  'annotation_column': 'broad'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = callback_add_region_annotation_to_quantification_frame(annotations_dict,
                                                                                    quantification_frame.to_dict(
                                                                                                  orient="records"),
                                                                                              "roi_1", None, False,
                                                                                              None)
    quantification_frame = pd.DataFrame(quantification_frame)
    assert "different" not in quantification_frame["ccramic_cell_annotation"].to_list()
    assert "different" in quantification_frame["broad"].to_list()

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

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

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'path',
                  'annotation_column': 'ccramic_cell_annotation'}

    annotations_dict = {"roi_1": {svgpath: annotation}}

    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    quantification_frame, serverside = callback_add_region_annotation_to_quantification_frame(annotations_dict,
                                        measurements.to_dict(orient="records"), "roi_1", mask_dict, True, "mask_1",
                                        sample_name='Dilution_series_1_1')
    quantification_frame = pd.DataFrame(quantification_frame)

    assert len(quantification_frame[
                   quantification_frame["ccramic_cell_annotation"] == "mature"]) == 2
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 1]["ccramic_cell_annotation"]) == ["None"]
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 403]["ccramic_cell_annotation"]) == ["mature"]

    annotations_dict = {'roi_1': {
        "{'points': [{'x': 534, 'y': 648}]}":
            {'title': 'fake_title', 'body': 'fake_body',
             'cell_type': 'mature', 'imported': False, 'type': 'point',
             'annotation_column': 'ccramic_cell_annotation'}
    }}

    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

    quantification_frame, serverside = callback_add_region_annotation_to_quantification_frame(annotations_dict,
                                                                                              measurements.to_dict(
                                                                                                  orient="records"),
                                                                                              "roi_1", mask_dict, True,
                                                                                              "mask_1",
                                                                                              sample_name='Dilution_series_1_1')
    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'ccramic_cell_annotation' in quantification_frame.columns


def test_basic_shape_removal_from_canvas():
    import plotly.graph_objects as go

    fig = go.Figure()

    # Create scatter trace of text labels
    fig.add_trace(go.Scatter(
        x=[1.5, 3],
        y=[2.5, 2.5],
        text=["Rectangle reference to the plot",
              "Rectangle reference to the axes"],
        mode="text",
    ))

    # Set axes properties
    fig.update_xaxes(range=[0, 4])
    fig.update_yaxes(range=[0, 4])
    assert len(fig['layout']['shapes']) == 0

    fig.add_shape(type="rect",
                  xref="x", yref="y",
                  x0=2.5, y0=0,
                  x1=3.5, y1=2,
                  line=dict(
                      color="RoyalBlue",
                      width=3,
                  ),
                  fillcolor="LightSkyBlue",
                  )

    assert len(fig['layout']['shapes']) == 1

    new_fig, warning = callback_remove_canvas_annotation_shapes(1, fig, {'autosize': True}, None)
    assert len(new_fig['layout']['shapes']) == 0

    new_fig, warning = callback_remove_canvas_annotation_shapes(1, fig, {'shapes': []}, None)
    assert isinstance(warning, dict)
    assert warning["error"] == "There are annotation shapes in the current layout. \n" \
                                "Switch to zoom or pan before removing the annotation shapes."
