import pytest
from ccramic.parsers.pixel_level_parsers import *

def test_basic_parser_tiff_to_dict(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "for_recolour.tiff")])
    assert isinstance(uploaded, tuple)
    uploaded_dict = uploaded[0]
    assert len(uploaded_dict['metadata']) > 0
    assert 'for_recolour+++slide0+++acq0' in uploaded_dict.keys()
    assert not 'experiment1' in uploaded_dict.keys()
    assert len(uploaded_dict['for_recolour+++slide0+++acq0'].keys()) == 1

    blending_dict = create_new_blending_dict(uploaded_dict)
    assert all([elem in ['#FFFFFF', None] for elem in \
                blending_dict['channel_1'].values()])

def test_basic_parser_fake_mcd(get_current_dir):
    with pytest.raises(ValueError):
        populate_upload_dict([os.path.join(get_current_dir, "fake_dataset.mcd")])


def test_basic_parser_from_mcd(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "query.mcd")])
    uploaded_dict = uploaded[0]
    assert 'query+++slide0+++Xylene' in uploaded_dict.keys()
    assert 'metadata' in uploaded_dict.keys()
    assert len(uploaded_dict['query+++slide0+++Xylene']) == 11
    # the values will all be none for the mcd because of lazy loadinfg
    assert all([value is None for value in uploaded_dict['query+++slide0+++Xylene'].values()])
    dataset_info = uploaded[-1]
    assert len(dataset_info['ROI']) == 6
    assert '11 markers' in dataset_info['Panel']
