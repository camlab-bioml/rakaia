import streamlit as st
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_clickable_images import clickable_images
import base64
from tempfile import NamedTemporaryFile
from streamlit_drawable_canvas import st_canvas
import cv2
import io
import re
import uuid
import tempfile

from io import BytesIO
from matplotlib import cm

if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""

st.set_page_config(layout="wide")

st.title("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry")

app_css = """
    <style>
        .main > div {
            padding-top: 2.5rem;
            padding-left: 10rem;
            padding-right: 10rem;
        }
            [data-testid="stSidebarNav"]::before {
                content: "Configure ccramic browser inputs";
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
         canvas   }
    </style>
"""
st.markdown(app_css, unsafe_allow_html=True)

dataframe_quant = None
jpeg = None
dataset = None
cell_image = None
IMAGE_HEIGHT = None
IMAGE_WIDTH = None
ASPECT_RATIO = None


def get_pasted_image(base_layer, top_layer, destination, aspect_ratio):

    top_layer.thumbnail((600, 600 * aspect_ratio), Image.Resampling.LANCZOS)

    # x, y = top_layer.size
    base_layer.paste(im, (0, 0), im)

    buffered = BytesIO()
    base_layer.save(buffered, format="PNG")
    img_data = buffered.getvalue()
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(img_data.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(img_data).decode()

    return (
            custom_css
            + f'<a download="{destination}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
    )



st.sidebar.title("Configure ccramic browser inputs")
# inputs declared in the sidebar can persist when changing tabs in the session
with st.sidebar:
    dataset = st.file_uploader("Select a quantification worksheet to view")
    permitted_dataset_formats = ['.csv', '.tsv']
    if dataset is not None and os.path.splitext(dataset.name)[1] in permitted_dataset_formats:
        dataframe_quant = pd.read_csv(dataset)
    first_tiff = st.file_uploader("Select a processed tiff file to view")


image_tab, quantification_tab, distribution_tab = st.tabs(["Multiplex Imaging", "Quantification", "Distribution"])
with image_tab:
    permitted_image_formats = ['.tif', '.png', '.jpeg']
    image_label = st.text_input("Annotate your tiff")
    if first_tiff is not None and os.path.splitext(first_tiff.name)[1] in permitted_image_formats:
        cell_image = Image.open(first_tiff)
        IMAGE_WIDTH, IMAGE_HEIGHT = cell_image.size
        ASPECT_RATIO = round(IMAGE_WIDTH/IMAGE_HEIGHT, 3)
        file_path_base = os.path.join(tempfile.gettempdir(), 'base_layer.png')
        cell_image.thumbnail((600, 600*ASPECT_RATIO), Image.Resampling.LANCZOS)
        cell_image.save(file_path_base)
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    drawing_mode = st.selectbox(
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
        )
    if drawing_mode == 'point':
        point_display_radius = st.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#eee")

    if st.session_state["button_id"] == "" or st.session_state["button_id"] is None:
        st.session_state["button_id"] = re.sub(
            "\d+", "", str(uuid.uuid4()).replace("-", "")
        )

    button_id = st.session_state["button_id"]

    custom_css = f""" 
            <style>
                #{button_id} {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: .25rem .75rem;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

    if cell_image is not None:
        data = st_canvas(drawing_mode=drawing_mode,
                     key="png_export",
                     stroke_width=stroke_width,
                     stroke_color=stroke_color,
                     background_image=cell_image if first_tiff is not None and
                                                    os.path.splitext(first_tiff.name)[
                                                        1] in permitted_image_formats else None,
                     update_streamlit=True,
                     height=600,
                     width=600*ASPECT_RATIO,
                     point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                     )

        if data is not None and data.image_data is not None and cell_image is not None:
            img_data = data.image_data
            im = Image.fromarray(img_data, mode="RGBA")

            file_path = os.path.join(tempfile.gettempdir(), f"{st.session_state['button_id']}_pasted.png")
            # base_layer_path = os.path.join(tempfile.gettempdir(), f"{st.session_state['button_id']}_base_layer.png")

            dl_link = get_pasted_image(cell_image, im, file_path, ASPECT_RATIO)

            st.markdown(dl_link, unsafe_allow_html=True)

    # print(Image.fromarray(canvas_result.image_data).paste(cell_image))

    """
    if data.image_data is not None:
        Image.fromarray(data.image_data).save("temp.png")
        Image.open("temp.png").save("temp_2.png")
        print(cell_image.paste(Image.open("temp.png"), box=(0, 0)))
        # cell_image.paste(Image.open("temp.png")).save("temp_2.png")
    """



with quantification_tab:

    if dataframe_quant is not None:
        gd = GridOptionsBuilder.from_dataframe(dataframe_quant if dataframe_quant is not None else None)
        gd.configure_pagination(enabled=True)
        gd.configure_side_bar()
        gd.configure_default_column(editable=True, groupable=True)
        gd.configure_selection(selection_mode="multiple", use_checkbox=False)
        gridoptions = gd.build()

        grid_table = AgGrid(
            dataframe_quant if dataframe_quant is not None else None,
            gridOptions=gridoptions,
            theme="material",
        )
        st.download_button(
            label="Download to CSV",
            data=pd.DataFrame(grid_table['data']).to_csv().encode("utf-8"),
            file_name="results.csv",
            mime="text/csv",
        )
        # sel_row = grid_table["selected_rows"]

with distribution_tab:
    hist_height = st.slider("plot height", 1, 25, 3)
    hist_width = st.slider("plot width", 1, 25, 1)
    if dataframe_quant is not None:
        distribution_choice = st.selectbox("Select a quantification distribution to view",
                                           options=dataframe_quant.columns)
        if distribution_choice is not None:
            hist = plt.figure(figsize=(hist_height, hist_width))
            plt.hist(dataframe_quant[distribution_choice].tolist())
            st.pyplot(hist)

