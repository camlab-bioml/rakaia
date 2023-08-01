import pytest

from ccramic.app.parsers.pixel_level_parsers import *

def test_basic_parser_tiff_to_dict(get_current_dir):
    uploaded = populate_upload_dict([os.path.join(get_current_dir, "for_recolour.tiff")])
    assert isinstance(uploaded, tuple)
    uploaded_dict = uploaded[0]
    assert len(uploaded_dict['metadata']) > 0
    assert 'experiment0' in uploaded_dict.keys()
    assert not 'experiment1' in uploaded_dict.keys()
    assert len(uploaded_dict['experiment0']['slide0']['acq0'].keys()) == 1


    blending_dict = create_new_blending_dict(uploaded_dict)
    assert all([elem in ['#FFFFFF', None] for elem in \
                blending_dict['experiment0']['slide0']['acq0']['channel_1'].values()])

def test_basic_parser_fake_mcd(get_current_dir):
    with pytest.raises(ValueError):
        populate_upload_dict([os.path.join(get_current_dir, "fake_dataset.mcd")])
