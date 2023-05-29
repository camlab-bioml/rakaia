# from imctools.converters import ome2analysis
# from imctools.converters import ome2histocat
# from imctools.converters import mcdfolder2imcfolder
# from imctools.converters import exportacquisitioncsv
import numpy as np
from PIL import Image
from PIL import ImageColor
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from skimage import data, draw
from scipy import ndimage
import numpy.ma as ma
from scipy.ndimage import gaussian_filter, median_filter


def get_luma(rbg):
    return 0.2126 * rbg[0] + 0.7152 * rbg[1] + 0.0722 * rbg[2]


def generate_tiff_stack(tiff_dict, tiff_list, colour_dict):
    # image = recolour_greyscale(tiff_dict[tiff_list[0]], colour_dict[tiff_list[0]])
    # for other in tiff_list[1:]:
    #     image = image + recolour_greyscale(tiff_dict[other], colour_dict[other]
    return Image.fromarray(sum([recolour_greyscale(tiff_dict[elem], colour_dict[elem]) for elem in tiff_list]))


def recolour_greyscale(array, colour):
    if colour not in ['#ffffff', '#FFFFFF']:
        image = Image.fromarray(array.astype(np.uint8))
        image = image.convert('RGB')
        red, green, blue = ImageColor.getcolor(colour, "RGB")

        # # array = np.array(image)
        # # print(array.shape)
        # # for i in range(3, 256):
        # #     array[(array[:, :, 0] == i) & (array[:, :, 1] == i) &
        # #             (array[:, :, 2] == i)] = [red*(i/255), green *(i/255), blue*(i/255)]
        #
        # pixels = image.load()
        #
        # for i in range(image.width):
        #     for j in range(image.height):
        #         if sum(pixels[i, j]) > 10 and pixels[i, j] is not None:
        #             # luma = get_luma(pixels[i, j])
        #         # transform = []
        #         # for col in [red, green, blue]:
        #         #     try:
        #         #         transform.append(int(col * (luma / 255)))
        #         #     except ZeroDivisionError:
        #         #         transform.append(0)
        #             val = pixels[i, j][0]
        #             pixels[i, j] = (int(red * (val / 255)), int(green * (val / 255)), int(blue * (val / 255)))
        #
        # return np.array(image)

        array = np.array(image)

        new_array = np.empty((array.shape[0], array.shape[1], 3))
        new_array[:, :, 0] = red
        new_array[:, :, 1] = green
        new_array[:, :, 2] = blue

        converted = new_array * (np.array(image) / 255)
        # print(converted)
        return converted.astype(np.uint8)

    else:
        image = Image.fromarray(array.astype(np.uint8))
        image = image.convert('RGB')
        return np.array(image).astype(np.uint8)


def convert_image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def read_back_base64_to_image(string):
    image_back = base64.b64decode(string)
    return Image.open(io.BytesIO(image_back))


# def fig_to_uri(in_fig, close_all=True, **save_args):
#     """
#     Save a figure as a URI
#     :param in_fig:
#     :return:
#     """
#     out_img = BytesIO()
#     in_fig.savefig(out_img, format='png', **save_args)
#     if close_all:
#         in_fig.clf()
#         plt.close('all')
#     out_img.seek(0)  # rewind file
#     encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
#     return "data:image/png;base64,{}".format(encoded)

def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


def get_area_statistics_from_rect(array, x_range_low, x_range_high, y_range_low, y_range_high):
    subset = array[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                          range(int(x_range_low), int(x_range_high), 1))]
    return np.average(subset), np.amax(subset), np.amin(subset)


def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def get_area_statistics_from_closed_path(array, svgpath):
    """
    Subset an array based on coordinates contained within a svg path drawn on the canvas
    """
    # https://dash.plotly.com/annotations?_gl=1*9dqxqk*_ga*ODM0NzUyNzQ3LjE2NjQyODUyNDc.*_ga_6G7EE0JNSC*MTY4MzU2MDY0My4xMDUuMS4xNjgzNTYyNDM3LjAuMC4w

    masked_array = path_to_mask(svgpath, array.shape)
    # masked_subset_data = ma.array(array, mask=masked_array)
    return np.average(array[masked_array]), np.amax(array[masked_array]), np.amin(array[masked_array])


def convert_to_below_255(array):
    return array if np.max(array) < 65000 else (array // 256).astype(np.uint8)


def resize_for_canvas(image, basewidth=400, return_array=True):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    if return_array:
        to_return = np.array(image.resize((basewidth, hsize), Image.Resampling.LANCZOS))
    else:
        to_return = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return to_return


def make_metadata_column_editable(column_name):
    # only allow the channel label column to be edited
    # return "Label" in column_name or column_name == "Channel Label"
    return column_name == "ccramic Label"


def filter_by_upper_and_lower_bound(array, lower_bound, upper_bound):
    """
    Filter an array by an upper and lower bound on the pixel values.
    Filter on the lower bound: removes any pixels less than the lower bound
    Filter on the upper bound: sets the upper bound as the new max intensity and scales all pixels
    relative to the new max.
    Example: original max intensity 255, new upper bound = 100. Scaling will be done to each pixel retained
    by multiplying by 255/100
    """
    # array = np.array(Image.fromarray(array).convert('L'))
    original_max = np.max(array)
    if None not in (original_max, upper_bound):
        scale_factor = float(original_max) / float(upper_bound)
    else:
        scale_factor = 1
    if lower_bound is None:
        lower_bound = 0
    else:
        lower_bound = float(lower_bound)
    array = np.where(array < lower_bound, 0, array)
    array = array * scale_factor
    return array


def pixel_hist_from_array(array):
    try:
        hist_data = np.hstack(array)
        max_hist = np.amax(array)
        hist = np.random.choice(hist_data, int(hist_data.shape[0] / 100)) if \
            hist_data.shape[0] > 2000000 else hist_data
        # add the largest pixel to ensure that hottest pixel is included in the distribution
        hist = np.concatenate(hist, int(max_hist))
        return go.Figure(px.histogram(hist, range_x=[min(hist), max(hist)]))
    except ValueError:
        return pixel_hist_from_array(np.array(Image.fromarray(array.astype(np.uint8)).convert('L')))


def apply_preset_to_array(array, preset):
    preset_keys = ['x_lower_bound', 'x_upper_bound', 'filter_type', 'filter_val']
    if isinstance(preset, dict) and all([elem in preset.keys() for elem in preset_keys]):
        array = filter_by_upper_and_lower_bound(array, preset['x_lower_bound'], preset['x_upper_bound'])
        if preset['filter_type'] == "median" and preset['filter_val'] is not None:
            array = median_filter(array, int(preset['filter_val']))
        elif preset['filter_val'] is not None:
            array = gaussian_filter(array, int(preset['filter_val']))
        return array


def apply_preset_to_blend_dict(blend_dict, preset_dict):
    """
    Populate the blend dict from a preset dict
    """
    assert all([key in blend_dict.keys() for key in preset_dict.keys()])
    for key, value in preset_dict.items():
        # do not change the color from a preset
        if key != "color":
            blend_dict[key] = value
    return blend_dict


def get_all_images_by_channel_name(upload_dict, channel_name):
    """
    Get all the images in a session dictionary from a channel name for the gallery view
    """
    images = {}
    for exp in list(upload_dict.keys()):
        if 'metadata' not in exp:
            for slide in upload_dict[exp].keys():
                for acq in upload_dict[exp][slide].keys():
                    for channel in upload_dict[exp][slide][acq].keys():
                        if channel == channel_name:
                            string = f"{exp}_{slide}_{acq}"
                            images[string] = upload_dict[exp][slide][acq][channel]
    return images
