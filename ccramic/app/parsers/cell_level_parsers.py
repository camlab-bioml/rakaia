import pandas as pd

def validate_incoming_measurements_csv(measurements_csv, current_image=None, validate_with_image=True,
                                       required_columns=['cell_id', 'x', 'y', 'x_max', 'y_max', 'area']):
    """
    Validate an incoming measurements CSV against the current canvas, and ensure that it has all of the required
    information columns
    """
    if not all([column in measurements_csv.columns for column in required_columns]):
        return None
    # check the measurement CSV against an image to ensure that the dimensions match
    elif validate_with_image and current_image is not None:
        if float(current_image.shape[0]) != float(measurements_csv['x_max'].max()) or \
            float(current_image.shape[1]) != float(measurements_csv['y_max'].max()):
            return None
        else:
            return measurements_csv
    else:
        return measurements_csv

def filter_measurements_csv_by_channel_percentile(measurements, percentile=0.999,
                                                  dropped_columns=['cell_id', 'x', 'y', 'x_max',
                                                                   'y_max', 'area', 'sample']):
    """
    Filter out the rows (cells) of a measurements csv (columns as channels, rows as cells) based on a pixel intensity
    threshold by percentile. Effectively removes any cells with "hot" pixels
    """
    try:
        measurements = pd.DataFrame(measurements).drop(dropped_columns, axis=1)
    except KeyError:
        pass
    query = ""
    quantiles = measurements.quantile(q=percentile)
    channel_index = 0
    for index, value in quantiles.items():
        query = query + f"`{index}` < {value}"
        if channel_index < len(quantiles) - 1:
            query = query + " & "
        channel_index += 1

    filtered = measurements.query(query, engine='python')
    return pd.DataFrame(filtered)
