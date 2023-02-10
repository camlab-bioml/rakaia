import streamlit as st
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

st.set_page_config(layout="wide")

st.title("ccramic: Cell-type Classification from Rapid Analysis of Multiplexed Imaging (mass) cytometry")

margins_css = """
    <style>
        .main > div {
            padding-top: 2.5rem;
            padding-left: 10rem;
            padding-right: 10rem;
        }
    </style>
"""
st.markdown(margins_css, unsafe_allow_html=True)

dataframe_quant = None
jpeg = None


def create_pdf(image, label):
    pdf = FPDF()
    pdf.add_page()
    pdf.image(image)
    pdf.cell(200, 10, txt=label,
                 ln=1, align='C')
    return pdf


image_tab, quantification_tab, distribution_tab = st.tabs(["Multiplex Imaging", "Quantification", "Distribution"])
with image_tab:
    first_tiff = st.file_uploader("Select a processed tiff file to view")
    permitted_image_formats = ['.tif', '.png', '.jpeg']
    image_label = st.text_input("Annotate your tiff")
    if first_tiff is not None and os.path.splitext(first_tiff.name)[1] in permitted_image_formats:
        cell_image = Image.open(first_tiff)
        st.image(cell_image)
        save_tiff = st.download_button("Save your annotated tif file", data=create_pdf(first_tiff.name, image_label))


with quantification_tab:
    dataset = st.file_uploader("Select a quantification worksheet to view")
    permitted_dataset_formats = ['.csv', '.tsv']
    if dataset is not None and os.path.splitext(dataset.name)[1] in permitted_dataset_formats:
        dataframe_quant = pd.read_csv(dataset)
        gd = GridOptionsBuilder.from_dataframe(dataframe_quant)
        gd.configure_pagination(enabled=True)
        gd.configure_side_bar()
        gd.configure_default_column(editable=True, groupable=True)
        gd.configure_selection(selection_mode="multiple", use_checkbox=False)
        gridoptions = gd.build()
        grid_table = AgGrid(
            dataframe_quant,
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
    hist_width = st.slider("plot width", 1, 25, 3)
    hist_height = st.slider("plot height", 1, 25, 1)
    if dataframe_quant is not None:
        distribution_choice = st.selectbox("Select a quantification distribution to view",
                                           options=dataframe_quant.columns)
        if distribution_choice is not None:
            hist = plt.figure(figsize=(hist_height, hist_width))
            plt.hist(dataframe_quant[distribution_choice].tolist())
            st.pyplot(hist)

