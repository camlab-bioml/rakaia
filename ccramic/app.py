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
            }
    </style>
"""
st.markdown(app_css, unsafe_allow_html=True)

dataframe_quant = None
jpeg = None
dataset = None


def create_pdf(image, label):
    pdf = FPDF()
    pdf.add_page()
    pdf.image(image)
    pdf.cell(200, 10, txt=label,
                 ln=1, align='C')
    return pdf


def save_annotated_image(canvas):
    cv2.imwrite(canvas.image_data)


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
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    drawing_mode = st.selectbox(
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
        )
    if drawing_mode == 'point':
        point_display_radius = st.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#eee")
    canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=cell_image if first_tiff is not None and
                                           os.path.splitext(first_tiff.name)[1] in permitted_image_formats else None,
            update_streamlit=True,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )

    # only writes the canvas changes not the background image
    # if canvas_result.image_data is not None:
        # cv2.imwrite(f"img.jpg",  canvas_result.image_data)

        # st.markdown(f"Image #{clickable} clicked" if clickable > -1 else "No image clicked")
        # save_tiff = st.download_button("Save your annotated image file", data=st.image(canvas_result.image_data))


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

