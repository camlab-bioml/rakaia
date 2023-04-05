# from imctools.converters import ome2analysis
# from imctools.converters import ome2histocat
# from imctools.converters import mcdfolder2imcfolder
# from imctools.converters import exportacquisitioncsv
import numpy as np
from PIL import Image
from PIL import ImageColor
import io
import base64


def get_luma(rbg):
    red, green, blue = rbg
    return 0.2126*red + 0.7152*red + 0.0722*blue


def generate_tiff_stack(tiff_dict, tiff_list, colour_dict):
    # image = recolour_greyscale(tiff_dict[tiff_list[0]], colour_dict[tiff_list[0]])
    # for other in tiff_list[1:]:
    #     image = image + recolour_greyscale(tiff_dict[other], colour_dict[other]
    return Image.fromarray(sum([recolour_greyscale(tiff_dict[elem], colour_dict[elem]) for elem in tiff_list]))


def recolour_greyscale(array, colour):
    image = Image.fromarray(array)
    image = image.convert('RGB')
    pixels = image.load()
    r, g, b = ImageColor.getcolor(colour, "RGB")
    for i in range(image.width):
        for j in range(image.height):
            if pixels[i, j] != (0, 0, 0):
                try:
                    luma = get_luma(pixels[i, j])
                    transform = []
                    for col in [r, g, b]:
                        try:
                            transform.append(int(col * (luma / 255)))
                        except ZeroDivisionError:
                            transform.append(0)
                    pixels[i, j] = (transform[0], transform[1], transform[2])
                except ZeroDivisionError:
                    pixels[i, j] = (0, 0, 0)

    return np.array(image)


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


def get_area_statistics(array, x_range_low, x_range_high, y_range_low, y_range_high):
    subset = array[np.ix_(range(int(y_range_low), int(y_range_high), 1),
                          range(int(x_range_low), int(x_range_high), 1))]
    return np.average(subset), np.amax(subset), np.amin(subset)
