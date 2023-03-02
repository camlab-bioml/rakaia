import streamlit as st
import os
from PIL import Image, ImageSequence
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
import json
import ast
import copy
import numpy as np
import plotly.figure_factory as ff

from io import BytesIO
from matplotlib import cm


# initialize important session state variables

session_states = ["button_id"]
for state in session_states:
    if state not in st.session_state:
        st.session_state[state] = ""

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
            + f'<a download="{destination}" id="{button_id}" href="data:file/txt;base64,{b64}">Export '
              f'annotated image</a><br></br>'
    )


st.sidebar.title("Configure ccramic browser inputs")
# inputs declared in the sidebar can persist when changing tabs in the session
with st.sidebar:
    dataset = st.file_uploader("Select a quantification worksheet to view")
    permitted_dataset_formats = ['.csv', '.tsv']
    if dataset is not None and os.path.splitext(dataset.name)[1] in permitted_dataset_formats:
        dataframe_quant = pd.read_csv(dataset)
    first_tiff = st.file_uploader("Select a processed tiff file to view")
    pre_annotated_json = st.file_uploader("Select a saved JSON to populate the canvas")

if "tabs" not in st.session_state:
    st.session_state["tabs"] = ["Multiplex Imaging", "Quantification", "Distribution"]
tabs = st.tabs(st.session_state["tabs"])

with tabs[0]:
    permitted_image_formats = ['.tif', '.png', '.jpeg']
    image_label = st.text_input("Annotate your tiff")
    if first_tiff is not None and os.path.splitext(first_tiff.name)[1] in permitted_image_formats:
        cell_image = Image.open(first_tiff)
        if os.path.splitext(first_tiff.name)[1] == ".tif":
            cell_image.load()
            # image_tab_0 = st.tabs(['image_tab_0'])
            if cell_image.n_frames > 1:
                index = 0
                tab_str = ""
                tab_assign_str = "st.tabs(["
                sub_images = []
                messages = {}
                for sub_image in ImageSequence.Iterator(cell_image):
                    sub_images.append(sub_image)
                    tab_str = tab_str + f"image_tab_{index}, " if index < (cell_image.n_frames - 1) else \
                        tab_str + f"image_tab_{index}"
                    tab_assign_str = tab_assign_str + f"'image_tab_{index}', " if index < (cell_image.n_frames - 1) else \
                        tab_assign_str + f"'image_tab_{index}'])"
                    messages[index] = f"This is tab number: {index}"
                    index += 1

                exec(f"{tab_str} = {tab_assign_str}")
                for i, page in enumerate(ImageSequence.Iterator(cell_image)):
                    img_byte_arr = io.BytesIO()
                    page.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    exec(f"with image_tab_{i}: st.image({img_byte_arr})")

        IMAGE_WIDTH, IMAGE_HEIGHT = cell_image.size
        ASPECT_RATIO = round(IMAGE_WIDTH/IMAGE_HEIGHT, 3)
        cell_image.thumbnail((600, 600*ASPECT_RATIO), Image.Resampling.LANCZOS)
        # cell_image.save(file_path_base)
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
        if pre_annotated_json is not None:
            pre_populate = pre_annotated_json.getvalue().decode("utf-8")
            replacements = {"null": "None", "false": 'False', "true": 'True'}

            copy = copy.deepcopy(pre_populate)
            for key, value in replacements.items():
                copy = copy.replace(key, value)

        data = st_canvas(drawing_mode=drawing_mode,
                         fill_color=bg_color,
                         initial_drawing=ast.literal_eval(copy) if pre_annotated_json is not None else None,
                     key="png_export",
                     stroke_width=stroke_width,
                     stroke_color=stroke_color,
                     background_color=bg_color,
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

            file_path = os.path.join(f"{st.session_state['button_id']}.png")
            # base_layer_path = os.path.join(tempfile.gettempdir(), f"{st.session_state['button_id']}_base_layer.png")

            dl_link = get_pasted_image(cell_image, im, file_path, ASPECT_RATIO)

            st.markdown(dl_link, unsafe_allow_html=True)

            if data.json_data is not None and len(data.json_data["objects"]) > 0:
                st.download_button(
                label="Download the canvas JSON",
                data=json.dumps(data.json_data, indent=2).encode('utf-8'),
                file_name="canvas.json",
                mime="application/json",
                )

with tabs[1]:

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

with tabs[2]:
    config_col, plot_col = st.columns([1,3])
    with config_col:
        hist_height = st.slider("plot height", 1, 25, 3)
        hist_width = st.slider("plot width", 1, 25, 1)
    with plot_col:
        if dataframe_quant is not None:
            distribution_choice = st.multiselect("Select a quantification distribution to view",
                                           options=dataframe_quant.columns,
                                             default=None)
            if distribution_choice is not None and len(distribution_choice) > 0:
                create_hist = [dataframe_quant[elem].sample(n=3000).tolist() for elem in distribution_choice]
                fig = ff.create_distplot(create_hist, distribution_choice)
                st.plotly_chart(fig, use_container_width=True)

