import pandas as pd
import os
from rakaia.inputs.metadata import metadata_association_plot

def test_metadata_association_plots(get_current_dir):
    measurements = pd.read_csv(os.path.join(get_current_dir, "cell_measurements.csv")).to_dict(orient="records")
    scatter = metadata_association_plot(measurements, '166Er_AR', '168Er_Ki67', '158Gd_GATA3')
    measurements = pd.DataFrame(measurements)
    assert scatter['data'][0]['type'] == 'scatter'
    # assert legend is numerical for scatter
    assert pd.Series(scatter['data'][0]['marker']['color']).equals(pd.Series(measurements['158Gd_GATA3']))
    scatter_2 = metadata_association_plot(measurements, '166Er_AR', '168Er_Ki67', 'sample')
    # legend exists with categorical
    assert scatter_2['data'][0]['showlegend']
    violin = metadata_association_plot(measurements, 'sample', '168Er_Ki67', 'sample')
    assert violin['data'][0]['type'] == 'violin'
    # legend used for categorical
    assert violin['data'][0]['showlegend']
    assert len(violin['data']) == len(measurements['sample'].value_counts())

    violin_2 = metadata_association_plot(measurements, 'sample', '168Er_Ki67', 'cell_id')
    # no legend if numerical legend passed to violin
    assert not violin_2['data'][0]['showlegend']
