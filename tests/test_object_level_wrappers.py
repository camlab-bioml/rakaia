import dash.exceptions
import pytest
import os
import tifffile
import pandas as pd
from dash.exceptions import PreventUpdate
from rakaia.callbacks.object_wrappers import (
    callback_remove_canvas_annotation_shapes,
    reset_annotation_import,
    AnnotationQuantificationMerge)
from rakaia.io.session import SessionServerside

def test_basic_callback_import_annotations_quantification_frame(get_current_dir, svgpath):
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    app_config = {'serverside_overwrite': True}
    bounds = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'zoom',
                  'annotation_column': 'rakaia_cell_annotation'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        measurements, "roi_1", None, False, None,
                                        config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)

    assert list(quantification_frame["rakaia_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                        (quantification_frame["y_max"] == 20)]) == ['mature']
    assert isinstance(serverside, SessionServerside)

    with pytest.raises(PreventUpdate):
        AnnotationQuantificationMerge(None, None, "roi_1", None, False, None).get_callback_structures()

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    bounds = {'x0': 826, 'x1': 836, 'y0': 12, 'y1': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'rect',
                  'annotation_column': 'rakaia_cell_annotation'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        measurements.to_dict(orient="records"), "roi_1", None, False,
                                        None, config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)

    assert list(quantification_frame["rakaia_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                                (quantification_frame["y_max"] == 20)]) == ['mature']

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'different', 'imported': False, 'type': 'rect',
                  'annotation_column': 'broad'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation}}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        quantification_frame.to_dict(orient="records"), "roi_1", None, False,
                                        None, config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert "different" not in quantification_frame["rakaia_cell_annotation"].to_list()
    assert "different" in quantification_frame["broad"].to_list()

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'path',
                  'annotation_column': 'rakaia_cell_annotation'}

    annotations_dict = {"roi_1": {svgpath: annotation}}

    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        measurements.to_dict(orient="records"), "roi_1", mask_dict, True, "mask_1",
                                        sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)

    assert len(quantification_frame[
                   quantification_frame["rakaia_cell_annotation"] == "mature"]) == 2
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 1]["rakaia_cell_annotation"]) == ['Unassigned']
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 403]["rakaia_cell_annotation"]) == ["mature"]

    with pytest.raises(dash.exceptions.PreventUpdate):
        quantification_frame, serverside = AnnotationQuantificationMerge(None,
                                    None,b"roi_1", None, False,
                                    None, sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()

    assert len(quantification_frame[
                   quantification_frame["rakaia_cell_annotation"] == "mature"]) == 2
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 1]["rakaia_cell_annotation"]) == ['Unassigned']
    assert list(quantification_frame[
                    quantification_frame["cell_id"] == 403]["rakaia_cell_annotation"]) == ["mature"]


    annotations_dict = {'roi_1': {
        "{'points': [{'x': 582, 'y': 465}]}":
            {'title': 'fake_title', 'body': 'fake_body',
             'cell_type': 'mature', 'imported': False, 'type': 'point',
             'annotation_column': 'rakaia_cell_annotation'}
    }}

    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                    measurements.to_dict(orient="records"), "roi_1", mask_dict, True,
                                    "mask_1", sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'rakaia_cell_annotation' in quantification_frame.columns
    assert 'mature' in list(quantification_frame['rakaia_cell_annotation'])

    gated_cell_tuple = (102, 154, 134, 201, 209, 244)
    # annotate using gated cell method
    annotations_dict = {'roi_1': {
        gated_cell_tuple:
            {'title': 'Unassigned', 'body': 'Unassigned',
             'cell_type': 'mature', 'imported': False, 'type': 'gate',
             'annotation_column': 'gating_test'}
    }}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                measurements.to_dict(orient="records"), "roi_1", mask_dict, True, "mask_1",
                                sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'gating_test' in quantification_frame.columns
    assert 'mature' in list(quantification_frame['gating_test'])
    # assert that the gated positions are offset by 1 (0-indexed)
    assert quantification_frame.index[quantification_frame['gating_test'] == 'mature'].tolist() == \
           [int(elem) - 1 for elem in sorted(list(gated_cell_tuple))]


def test_basic_annotation_dict_without_quantification_frame(get_current_dir):

    annotations_dict = {'roi_1': {
        "{'points': [{'x': 582, 'y': 465}]}":
            {'title': 'fake_title', 'body': 'fake_body',
             'cell_type': 'mature', 'imported': False, 'type': 'point',
             'annotation_column': 'rakaia_cell_annotation'}
    }}

    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}
    app_config = {'serverside_overwrite': True}
    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        None, "roi_1", mask_dict, True, "mask_1", sample_name='Dilution_series_1_1',
                                        config=app_config).get_callback_structures()
    assert quantification_frame is None
    assert not serverside.value['roi_1']["{'points': [{'x': 582, 'y': 465}]}"]['imported']

    quantification_frame, serverside = AnnotationQuantificationMerge(serverside.value,
                    None, "roi_1", mask_dict, True, "mask_1", sample_name='Dilution_series_1_1',
                    config=app_config, remove=True).get_callback_structures()
    assert not serverside.value['roi_1']


def test_basic_callback_remove_annotations_quantification_frame(get_current_dir):
    """
    Test that removing an annotation sets the column back to the defaults
    Here, remove the first annotation an keep the last one
    """
    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))
    app_config = {'serverside_overwrite': True}
    bounds = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'mature', 'imported': False, 'type': 'zoom',
                  'annotation_column': 'rakaia_cell_annotation', 'id': 'annot_1'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation,
                                  "{'points': [{'x': 582, 'y': 465}]}":
                                      {'title': 'fake_title', 'body': 'fake_body',
                                       'cell_type': 'mature', 'imported': False, 'type': 'point',
                                       'annotation_column': 'rakaia_cell_annotation', 'id': 'annot_2'}
                                  }}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                        measurements.to_dict(orient="records"), "roi_1", None, False, None,
                                        config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)

    assert list(quantification_frame["rakaia_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                        (quantification_frame["y_max"] == 20)]) == ['mature']

    # Remove the most recent annotation

    quantification_frame, serverside = AnnotationQuantificationMerge(serverside.value,
                                    quantification_frame.to_dict(orient="records"), "roi_1", None, False, None,
                                    config=app_config, remove=True, indices_remove=[0]).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert list(quantification_frame["rakaia_cell_annotation"][(quantification_frame["x_max"] == 836) &
                                                                (quantification_frame["y_max"] == 20)]) == ['Unassigned']

def test_basic_callback_remove_annotations_quantification_frame_2(get_current_dir):
    """
    Test that removing an annotation sets the column back to the defaults
    Here, remove the last annotation by index and keep the first
    """
    app_config = {'serverside_overwrite': True}
    bounds = {'xaxis.range[0]': 826, 'xaxis.range[1]': 836, 'yaxis.range[0]': 12, 'yaxis.range[1]': 21}

    annotation = {'title': 'fake_title', 'body': 'fake_body',
                  'cell_type': 'first', 'imported': False, 'type': 'zoom',
                  'annotation_column': 'rakaia_cell_annotation', 'id': 'annot_1'}

    annotations_dict = {"roi_1": {tuple(sorted(bounds.items())): annotation,
                                  "{'points': [{'x': 582, 'y': 465}]}":
                                      {'title': 'fake_title', 'body': 'fake_body',
                                       'cell_type': 'mature', 'imported': False, 'type': 'point',
                                       'annotation_column': 'rakaia_cell_annotation', 'id': 'annot_2'}
                                  }}
    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
                                    measurements.to_dict(orient="records"), "roi_1", mask_dict, True,
                                    "mask_1", sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'rakaia_cell_annotation' in quantification_frame.columns
    assert 'mature' in list(quantification_frame['rakaia_cell_annotation'])
    assert list(quantification_frame['rakaia_cell_annotation'].value_counts().index) == ['Unassigned', 'first', 'mature']

    quantification_frame, serverside = AnnotationQuantificationMerge(serverside.value,
                                    quantification_frame.to_dict(orient="records"), "roi_1", mask_dict, True,
                                    "mask_1", sample_name='Dilution_series_1_1', config=app_config, remove=True,
                                    indices_remove=[1]).get_callback_structures()
    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'rakaia_cell_annotation' in quantification_frame.columns
    assert not 'mature' in list(quantification_frame['rakaia_cell_annotation'])
    assert list(quantification_frame['rakaia_cell_annotation'].value_counts().index) == ['Unassigned', 'first']

def test_basic_callback_remove_annotations_quantification_frame_3(get_current_dir):
    """
    Test that removing an annotation sets the column back to the defaults
    Here, remove the last annotation without an index using gating
    """
    app_config = {'serverside_overwrite': True}
    mask_dict = {"mask_1": {"raw": tifffile.imread(os.path.join(get_current_dir, "mask.tiff"))}}

    measurements = pd.read_csv(os.path.join(get_current_dir, "measurements_for_query.csv"))

    gated_cell_tuple = (102, 154, 134, 201, 209, 244)
    # annotate using gated cell method
    annotations_dict = {'roi_1': {
        gated_cell_tuple:
            {'title': 'Unassigned', 'body': 'Unassigned',
             'cell_type': 'mature', 'imported': False, 'type': 'gate',
             'annotation_column': 'gating_test', 'mask_selection': 'mask', 'id': 'gating_1'}
    }}

    quantification_frame, serverside = AnnotationQuantificationMerge(annotations_dict,
            measurements.to_dict(orient="records"), "roi_1", mask_dict, True, "mask_1",
            sample_name='Dilution_series_1_1', config=app_config).get_callback_structures()

    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'gating_test' in quantification_frame.columns
    assert 'mature' in list(quantification_frame['gating_test'])
    # assert that the gated positions are offset by 1 (0-indexed)
    assert quantification_frame.index[quantification_frame['gating_test'] == 'mature'].tolist() == \
           [int(elem) - 1 for elem in sorted(list(gated_cell_tuple))]

    quantification_frame, serverside = AnnotationQuantificationMerge(serverside.value,
                    quantification_frame.to_dict(orient="records"), "roi_1", mask_dict, True, "mask_1",
                    sample_name='Dilution_series_1_1', config=app_config,
                    remove=True, indices_remove=[0, 1, 2]).get_callback_structures()

    quantification_frame = pd.DataFrame(quantification_frame)
    assert 'gating_test' in quantification_frame.columns
    assert not 'mature' in list(quantification_frame['gating_test'])
    assert quantification_frame.index[quantification_frame['gating_test'] == 'mature'].tolist() == []

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

def test_reset_annotations_import():
    annot_dict = {"roi_1": {"annot_1": {"imported": True}, "annot_2": {"imported": True}, "annot_3": {"imported": True}}}
    assert all([elem['imported'] for elem in annot_dict['roi_1'].values()])
    annot_dict_serverside = reset_annotation_import(annot_dict, "roi_1", {"serverside_overwrite": True})
    assert all([not elem['imported'] for elem in annot_dict_serverside.value['roi_1'].values()])

    # for ROIs not specified as the current one, do not reimport
    annot_mismatch = {"roi_2": {"annot_1": {"imported": True},
                                "annot_2": {"imported": True}, "annot_3": {"imported": True}}}
    annot_reset_mismatch = reset_annotation_import(annot_mismatch, "roi_1", {"serverside_overwrite": True}, False)
    assert annot_mismatch == annot_reset_mismatch
