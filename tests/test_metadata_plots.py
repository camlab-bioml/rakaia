import pandas as pd
import os
from rakaia.inputs.metadata import metadata_association_plot

def test_metadata_association_plots(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv"))
    scatter = metadata_association_plot(measurements, '166Er_AR', '168Er_Ki67', 'cell_id')
    assert scatter['data'][0]['type'] == 'scatter'
    # assert no legend if there is no grouping
    assert not scatter['data'][0]['showlegend']
    scatter_2 = metadata_association_plot(measurements, '166Er_AR', '168Er_Ki67', 'sample')
    # assert no legend if there is no grouping
    assert scatter_2['data'][0]['showlegend']
    violin = metadata_association_plot(measurements, 'sample', '168Er_Ki67', 'sample')
    assert violin['data'][0]['type'] == 'violin'
    assert violin['data'][0]['showlegend']
