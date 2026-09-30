"""
Module related to functions and classes for processing WSI patches and enabling API queries
"""
import io
import copy
from http.client import HTTPException
from typing import Union
from pathlib import Path
import textwrap
import pandas as pd
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import fisher_exact

# IMP: need this to map the tissue description from hist2query to the full metadata
# projects because the description is not held there
TCGA_DISEASE_TO_PROJ_CODE = {
    "Adrenocortical carcinoma": "ACC",
    "Bladder Urothelial Carcinoma": "BLCA",
    "Breast invasive carcinoma (IDC)": "BRCA",
    "Breast invasive carcinoma (Other)": "BRCA",
    "Cervical squamous cell carcinoma and endocervical adenocarcinoma": "CESC",
    "Cholangiocarcinoma": "CHOL",
    "Colon adenocarcinoma": "COAD",
    "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma": "DLBC",
    "Esophageal carcinoma": "ESCA",
    "Glioblastoma multiforme": "GBM",
    "Head and Neck squamous cell carcinoma": "HNSC",
    "Kidney Chromophobe": "KICH",
    "Kidney renal clear cell carcinoma": "KIRC",
    "Kidney renal papillary cell carcinoma": "KIRP",
    "Brain Lower Grade Glioma": "LGG",
    "Liver hepatocellular carcinoma": "LIHC",
    "Lung adenocarcinoma": "LUAD",
    "Lung squamous cell carcinoma": "LUSC",
    "Mesothelioma": "MESO",
    "Ovarian serous cystadenocarcinoma": "OV",
    "Pancreatic adenocarcinoma": "PAAD",
    "Pheochromocytoma and Paraganglioma": "PCPG",
    "Prostate adenocarcinoma": "PRAD",
    "Rectum adenocarcinoma": "READ",
    "Sarcoma": "SARC",
    "Skin Cutaneous Melanoma": "SKCM",
    "Stomach adenocarcinoma": "STAD",
    "Testicular Germ Cell Tumors": "TGCT",
    "Thyroid carcinoma": "THCA",
    "Thymoma": "THYM",
    "Uterine Corpus Endometrial Carcinoma": "UCEC",
    "Uterine Carcinosarcoma": "UCS",
    "Uveal Melanoma": "UVM",
}

# use these to map different project codes from tcga to cioportal
# if not in this list, then cbioportal uses the lower case project code for a gdc study
# i.e. https://www.cbioportal.org/study/clinicalData?id=brca_tcga_gdc
CBIOPORTAL_TCGA_CODE_MAP = {
    "PCPG": "mnet",
    "LGG": "difg",
    "MESO": "plmeso",
    "OV": "hgsoc",
    "SARC": "soft_tissue",
    "TGCT": "nsgct",
    "KICH": "chrcc",
    "KIRC": "ccrcc",
    "KIRP": "prcc",
    "LIHC": "hcc",
    "DLBC": "dlbclnos",
    "THCA": "thpa",
    "UVM": "um"}

# Default column definitions for the TCGA UNI search results shown in dash ag grid
TCGA_UNI_COL_DEFS = [{"field": "tissue", "rowGroup": True, "hide": True}, {"field": "slide", "rowGroup": True, "hide": True},
                    {"field": "x"}, {"field": "y"},
                    {"field": "url", "cellRenderer": "LinkRenderer"}, {"field": "similarity"},
                    {"field": "open", "headerName": "Open patch in slide viewer", "cellRenderer": "OpenSlideButton",
                    "cellRendererParams": {"className": "open-slide-btn"},
                    # static label shown on every button in this column
                    "valueGetter": {"function": "'Open'"}, "sortable": False, "filter": False}]

def crop_aspect_ratio(x0: Union[int, float], x1: Union[int, float], y0: Union[int, float], y1: Union[int, float]):
    """
    Set the crop aspect ratio as the width / height
    """
    return float((x1 - x0) / (y1 - y0))


def wsi_crop(image: Union[Path, str, np.ndarray, None],
             bounds: Union[list, None]=None,
             return_sampled: bool=True,
             patch_out_size: int=224):
    """
    Generate a crop of a WSI image processed through pyvips. Assumes that the bounds array is in the
    format `[x0, x1, y0, y1]`.
    If `return_subsample` is used, specify the size (i.e. 224 works for UNI patch embeddings)
    """
    import pyvips
    try:
        x0, x1, y0, y1 = bounds
        crop = pyvips.Image.new_from_file(image, access="sequential") if not \
            isinstance(image, np.ndarray) else pyvips.Image.new_from_array(image, interpretation='rgb')
        crop = crop.crop(x0, y0, x1 - x0, y1 - y0).numpy().astype(np.uint8)
        # drop alpha channel if present, often from svs
        if (len(crop.shape) == 3) and crop.shape[2] == 4: crop = crop[:, :, :3]
        return np.array(Image.fromarray(crop).resize((int(patch_out_size * float(crop_aspect_ratio(x0, x1, y0, y1))), patch_out_size),
             resample=Image.Resampling.LANCZOS)) if (return_sampled and patch_out_size > 0) else crop
    except (pyvips.Error, TypeError, KeyError): pass
    return None

def serialize_crop(crop: Union[np.array, np.ndarray, None]=None):
    """
    Serialize the WSI crop into compressed bytes for a POST request
    """
    if crop is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=np.stack(crop))
        return buffer.getvalue()
    return None

def tcga_resp_to_table(resp: Union[dict, None]=None):
    """
    Format the TCGA UNI POST response into a table (record-oriented) for viewing
    """
    if resp is not None and all(elem in resp.keys() for elem in ('hits', 'url')):
        hits_frame = pd.DataFrame(resp['hits'])
        hits_frame['url'] = "NA"
        if resp['url'] and isinstance(resp['url'], dict):
            hits_frame['url'] = hits_frame['slide'].map(resp['url'])
        return hits_frame.to_dict(orient="records")
    return None

def set_query_host(api_host: str="localhost",
                   api_port: int=6000):
    """
    Set the host and port for hist2query. Accepts localhost + port or a URL
    """
    if str(api_host).startswith("http") or str(api_host).startswith("https"): return api_host
    return f"http://{api_host}:{api_port}"

def tcga_uni_request(crop: Union[np.ndarray, np.array, None]=None,
                            api_host: str="localhost",
                            api_port: int=7000,
                            k_search: int=10,
                            return_url: bool=True,
                            endpoint: str="search",
                            return_processed: bool=True):
    """
    Format the TCGA UNI POST request to send to hist2query
    """
    if crop is not None:
        response = requests.post(f"{set_query_host(api_host, api_port)}/{endpoint}",
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"k": k_search, "url": return_url}, timeout=300)
        resp = handle_request_error(response)
        return tcga_resp_to_table(resp) if return_processed else resp
    return None

def handle_request_error(response: requests.request):
    """
    Handle the HTTP error and exceptions from hist2query
    """
    try:
        response.raise_for_status()
    except (requests.exceptions.HTTPError, HTTPException) as e:
        raise HTTPException(response.status_code, e.response.json().get("detail", "Error"))
    return response.json()

def prism2_chat_request(crop: Union[np.ndarray, np.array, None]=None,
                            api_host: str="localhost",
                            api_port: int=7000,
                            endpoint: str="chat",
                            question: str="What type of tissue is this?"):
    if crop is not None:
        response = requests.post(f"{set_query_host(api_host, api_port)}/{endpoint}",
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"question": question, "raw_scores_binary": False,
                                       "max_token_response": 100}, timeout=300)
        resp = handle_request_error(response)
        return resp['response'][0] if isinstance(resp['response'], list) else str(resp['response'])
    return None

def format_col_ag_groupings(use_grouping: bool=True):
    """
    format the col groupings
    """
    new_col_defs = copy.deepcopy(TCGA_UNI_COL_DEFS)
    for col in new_col_defs:
        if "rowGroup" in col:
            col["rowGroup"] = use_grouping
            col["hide"] = use_grouping
    return new_col_defs

def tile_dimension_labels(max_dim: int=10):
    """
    Set the `dcc.Dropdown` options and labels for the hist2query tile number downsample
    """
    return [{"label": f"{val}x{val}", "value": val} for val in range(1, int(max_dim + 1))]

def hist2query_tissue_list(query_results: Union[list, dict, None],
                    category: str="tissue"):
    """
    Return a list of the tissue types in the hist2query results for the clinical metadata dropdown filter
    """
    if query_results is not None:
        return list(pd.DataFrame(query_results)[category].unique())
    return []

def hist2query_pie_chart(query_results: Union[list, dict, None],
                    category: str="tissue"):
    """
    Generate a pie chart of the hist2query results by tissue or project
    """
    if None not in (query_results, category):
        query_results_plot = pd.DataFrame(query_results)
        grouping_col = category if category in query_results_plot.columns else "project"
        query_results_plot[grouping_col] = query_results_plot[grouping_col].apply(
            lambda x: "<br>".join(textwrap.wrap(x, width=25)))
        fig = go.Figure(px.pie(query_results_plot[grouping_col]
                               .value_counts(dropna=False)
                               .rename_axis("Tissue Type")
                               .reset_index(name="Count"),
                               names="Tissue Type",
                               values="Count",
                               title=f"Query {str(grouping_col)} distribution"))
        fig.update_layout(autosize=True, margin=dict(l=0, r=50, t=50, b=0),
                          legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1))
        return fig.to_dict()
    return None

_GDC_SlIDE_TEMPLATE = (Path(__file__).parent / "../templates" / "gdc.html").read_text()

TCGA_CLINICAL_METADATA_PATH = (Path(__file__).resolve().parent / "tcga_patient_metadata.parquet")
TCGA_DROPDOWN_COLS_IGNORE = ['bcr_patient_barcode']

def set_tcga_metadata_options():
    """
    Set the TCGA metadata column dropdown options
    """
    return [elem for elem in list(pd.read_parquet(TCGA_CLINICAL_METADATA_PATH).columns) if
            elem not in TCGA_DROPDOWN_COLS_IGNORE]


def gdc_slide_iframe(url: str, file_id: str, x: float, y: float, n: float = 1250) -> str:
    """
    Generate a GDC slide OSD viewer for a specific GDC-hosted slide with a patch highlighted by coordinates
    and a bounding box. Compatible with hist2query patch results
    """
    return (_GDC_SlIDE_TEMPLATE
        .replace("__URL__", str(url))
        .replace("__FILE_ID__", str(file_id))
        .replace("__X__", repr(float(x)))
        .replace("__Y__", repr(float(y)))
        .replace("__N__", repr(float(n))))

def hist2query_clinical_bar_plot(query_results: Union[list, pd.DataFrame],
                                 metadata_var: Union[str, None]=None,
                                 subset_tissue_groups: Union[list, None]=None,
                                 min_patch_count_per_patient: Union[int, None]=None,
                                 tissue_col_identifier: str="tissue",
                                 patient_col_identifier: str="bcr_patient_barcode"):
    """
    A bar plot of hist2query results per patient coloured by a metadata variable.
    Patients are ordered descending by the number of query patches
    """
    if metadata_var is not None and query_results is not None:
        result_frame = pd.DataFrame(query_results)
        tot_result = len(result_frame)
        clinical_meta = pd.read_parquet(TCGA_CLINICAL_METADATA_PATH)
        if subset_tissue_groups is not None and (isinstance(subset_tissue_groups, list) and len(subset_tissue_groups) > 0):
            result_frame = result_frame[result_frame[tissue_col_identifier].isin(subset_tissue_groups)]
        patches_in_view = len(result_frame)
        result_frame[patient_col_identifier] = [str(elem).split("-01Z")[0] for elem in result_frame['slide']]
        merged = result_frame.merge(clinical_meta, on=patient_col_identifier, how='inner')
        merged = merged.fillna("Not provided")
        patient_counts = (merged.groupby(patient_col_identifier)
                          .agg(patch_count=(patient_col_identifier, "size"),
                               **{str(metadata_var): (str(metadata_var), "first")})
                          .reset_index().sort_values("patch_count", ascending=False))
        if min_patch_count_per_patient is not None:
            patient_counts = patient_counts[patient_counts['patch_count'] >= min_patch_count_per_patient]
            patches_in_view = int(patient_counts['patch_count'].sum())

        counts = patient_counts[str(metadata_var)].value_counts()
        proportions = patient_counts[str(metadata_var)].value_counts(normalize=True).round(3)
        metadata_prop = (pd.concat([counts, proportions], axis=1, keys=["Counts", "Proportion"])
            .reset_index().rename(columns={str(metadata_var): 'Value'}))

        fig = px.bar(patient_counts, x=patient_col_identifier,
                     y="patch_count", color=str(metadata_var),
                     category_orders={patient_col_identifier: patient_counts[patient_col_identifier].tolist()},
                     title=f"Patients by {str(metadata_var)}, ({len(patient_counts)} patients, {patches_in_view}/{tot_result} query patches)")

        fig.update_layout(xaxis_title="Patient", yaxis_title="Number of result patches")
        return fig, metadata_prop.to_dict(orient="records"), patient_counts.to_dict(orient="records")
    return None, None, None

# exclude these values from the patient enrichment computation
TCGA_METADATA_VALS_EXCLUDE = ["[Discrepancy]", "[Not Available]", "Stage X", "NA", "None", "",
                              '[Unknown]', 'Unknown', '[Not Applicable]', 'GX', '[Not Evaluated]', None, 'Rx']
ENRICHMENT_COLS = [{'id': p, 'name': p, 'editable': False} for p in ['Value', 'Enrichment', 'Odds Ratio', 'P-value']]

def hist2query_patient_enrichment(patient_props: Union[list, pd.DataFrame, None]=None,
                                 metadata_var: Union[str, None]=None,
                                 subset_tissue_groups: Union[list, None]=None,
                                 tissue_col_identifier: str = "type",
                                 patient_col_identifier: str = "bcr_patient_barcode"):
    """
    Using a hist2query patient distribution table by metadata variable,
    compute the enrichment per category using Fisher's exact test.
    Answers: is this particular patient metadata variable more enriched in the query
    as opposed to the full TCGA metadata?
    """
    if patient_props is not None and not (isinstance(patient_props, list) and not patient_props) and metadata_var is not None:
        clinical_meta = pd.read_parquet(TCGA_CLINICAL_METADATA_PATH)
        if subset_tissue_groups is not None and (isinstance(subset_tissue_groups, list) and len(subset_tissue_groups) > 0):
            clinical_meta = clinical_meta[clinical_meta[tissue_col_identifier].isin(
            set([TCGA_DISEASE_TO_PROJ_CODE[tissue] for tissue in subset_tissue_groups]))]
        full_counts = clinical_meta.groupby(metadata_var)[patient_col_identifier].nunique()
        query_counts = pd.DataFrame(patient_props).set_index("Value")["Counts"]

        query_counts = query_counts.reindex(full_counts.index, fill_value=0)

        # TODO: should the non-descriptive columns be excluded?
        full_counts = full_counts.drop(labels=TCGA_METADATA_VALS_EXCLUDE, errors="ignore")
        query_counts = query_counts.drop(labels=TCGA_METADATA_VALS_EXCLUDE, errors="ignore")

        results = []
        for category in full_counts.index:
            query_count = query_counts[category]
            background_count = full_counts[category]

            query_not_category = query_counts.sum() - query_count
            non_query_category = background_count - query_count
            non_query_not_category = (
                    full_counts.sum()
                    - query_counts.sum()
                    - non_query_category)

            table = [[query_count, query_not_category],
                [non_query_category, non_query_not_category]]

            odds_ratio, pvalue = fisher_exact(table, alternative="greater")

            query_proportion = query_count / query_counts.sum()
            background_proportion = background_count / full_counts.sum()

            results.append({
                "Value": category,
                # "Query Count": query_count,
                # "Background Count": background_count,
                #"Query Proportion": query_proportion,
                #"Background Proportion": background_proportion,
                "Enrichment": query_proportion / background_proportion,
                "Odds Ratio": odds_ratio,
                "P-value": pvalue})

        results = pd.DataFrame(results).round(3)
        if not results.empty:
            # IMP: compute the test statistics for all categories, but only show the ones present in the query
            results = results[results['Value'].isin(list(pd.DataFrame(patient_props)['Value'].unique()))]
            cols = [{'id': p, 'name': p, 'editable': False} for p in list(results.columns)]
            return results.to_dict(orient="records"), cols
        return pd.DataFrame({}).to_dict(orient="records"), ENRICHMENT_COLS
    return pd.DataFrame({}).to_dict(orient="records"), ENRICHMENT_COLS

def cbioportal_patient_urls(patient_counts: Union[list, pd.DataFrame, None]=None,
                            tissue_col_identifier: str = "type",
                            patient_col_identifier: str = "bcr_patient_barcode",
                            patch_col: str = "patch_count"):
    """
    Generate a data table of the cBioPortal patient URLs for the TCGA queries. Maps the TCGA patient ID
    to a link in the matched TCGA GDC project hosted on cBioPortal
    """
    if patient_counts is not None and not (isinstance(patient_counts, list) and not patient_counts):
        clinical_meta = pd.read_parquet(TCGA_CLINICAL_METADATA_PATH)
        patient_counts = pd.DataFrame(patient_counts)[[patient_col_identifier, patch_col]].merge(
            clinical_meta[[patient_col_identifier, tissue_col_identifier]], on=patient_col_identifier, how="left")
        patient_counts[patient_col_identifier] = ("[" + patient_counts[patient_col_identifier] + "]" +
                    "(https://www.cbioportal.org/patient?studyId=" + patient_counts[tissue_col_identifier].map(
                    lambda x: CBIOPORTAL_TCGA_CODE_MAP.get(x, x.lower())) + "_tcga_gdc&caseId="
                    + patient_counts[patient_col_identifier] + ")")
        return patient_counts.rename(columns={patient_col_identifier: "Patient"}).drop(
            columns=tissue_col_identifier).to_dict(orient="records")
    return None
