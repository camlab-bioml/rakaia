

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
        if int(current_image.shape[0]) != int(measurements_csv['x_max'].argmax()) or \
            int(current_image.shape[1]) != int(measurements_csv['y_max'].argmax()):
            return None
        else:
            pass
    else:
        return measurements_csv
