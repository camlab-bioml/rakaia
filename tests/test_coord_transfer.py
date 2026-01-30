import os
import numpy as np
from rakaia.register.coordinates import WSICanvasAffineCoordTransfer

def test_non_visium_coord_transfer(get_current_dir):
    bounds = {'xaxis.range[0]': 36.7, 'xaxis.range[1]': 281.0,
              'yaxis.range[0]': 31.8, 'yaxis.range[1]': 147.1}
    string_coords = (WSICanvasAffineCoordTransfer(bounds,
                    os.path.join(get_current_dir, 'melanoma_xenium_subset.h5ad'),
                    os.path.join(get_current_dir, 'melanoma_xenium_transformation.csv')).
                    process_coordinates())
    x, y, width, height = tuple([float(elem) for elem in string_coords.split(",")])
    assert y > x
    assert height > width

    # test with a non-Anndata type file and perform no transformation
    no_transform = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]])

    string_coords = (WSICanvasAffineCoordTransfer(bounds,
                    os.path.join(get_current_dir, 'query.mcd'),
                    no_transform).process_coordinates(scaling_factor=1))
    x, y, width, height = tuple([float(elem) for elem in string_coords.split(",")])

    assert float(x) == float(bounds['xaxis.range[0]'])
    assert float(y) == float(bounds['yaxis.range[0]'])
    assert int(width) == int(float(bounds['xaxis.range[1]']) - float(bounds['xaxis.range[0]']))
    assert int(height) == int(float(bounds['yaxis.range[1]']) - float(bounds['yaxis.range[0]']))
