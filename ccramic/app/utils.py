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
        image = Image.fromarray(array)
        image = image.convert('RGB')
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
    return 100 * np.average(subset) / np.max(array), 100 * np.amax(subset) / np.max(array), \
           100 * np.amin(subset) / np.max(array)


def convert_to_below_255(array):
    return array if np.max(array) <= 255 else (array // 256).astype(np.uint8)


def resize_for_canvas(image, basewidth=400):
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    return image.resize((basewidth, hsize), Image.Resampling.LANCZOS)


def make_metadata_column_editable(column_name):
    # only allow the channel label column to be edited
    return "Label" in column_name or column_name == "Channel Label"
