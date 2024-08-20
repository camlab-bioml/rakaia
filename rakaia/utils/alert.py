import pathlib
from typing import Union
from pydantic import BaseModel

class DataImportTour(BaseModel):
    """
    Contains the steps used for the dash tour component for data import assistance in the `steps` atttribute

    :return: None
    """
    steps: list = [{'selector': '[id="upload-image"]',
                    'content': "Option 1: Upload your images (.mcd, .tiff, etc.) using drag and drop. Should"
                               " be used only for datasets < 2GB or if the app deployment is public/shared, "
                               "as the component creates a temporary copy of the file contents."},
                {'selector': '[id="read-filepath"]',
                'content': "Option 2: For large datasets (> 2GB) on local deployments, "
                           "copy and paste either a filepath or directory and "
                               "read files directly by selecting Import local. Does not duplicate any data."},
                {'selector': '[id="show-dataset-info"]',
                'content': 'View a list of imported datasets and regions of interest (ROIs).'
                           ' Multiple ROIs, files, and/or filetypes can be imported into the same session, '
                           'provided that the biomarker panel is the same across all ROIs.'},
                {'selector': '[id="data-collection"]',
                'content': 'Select an ROI from the dropdown menu to populate the image gallery'
                               ' and begin image analysis'},
                {'selector': '[id="annotation-canvas"]',
                    'content': 'Create a multiplexed image in the canvas by selecting channels/biomarkers\n'
                               ' from the Channel selection dropdown.\n'},
                   {'selector': '[id="rakaia-documentation"]',
                    'content': 'Visit the documentation for a full comprehensive user guide.'}]

class AlertMessage(BaseModel):
    """
    Contains hash containing a simple string alert message display in the `error warning modal in the `warnings` attribute

    :return: None
    """
    warnings: dict = {"blend_json_success": "Blend parameters successfully updated from JSON.",
                      "invalid_path": "Invalid filepath ir directory provided. Please verify the following: \n\n" \
                                        "- That the file path provided is a valid local file. \n" \
                                        "- That the directory provided exists and contains imaging files. \n"
                                        "- If running using Docker or a web version, " \
                                        "local file paths will not be available.",
                      "multiple_filetypes": "Warning: Multiple different file types were detected on upload. " \
                                        "This may cause problems during analysis. For best performance, " \
                                        "it is recommended to analyze datasets all from the same file type extension " \
                                        "and ensure that all imported datasets share the same panel.\n\n",
                      "json_update_success": "Blend parameters successfully updated from JSON.",
                      "json_update_error": "Error: the blend parameters uploaded from JSON do not " \
                                        "match the current session panel/channel list. The update did not occur. \n"
                            "Note that different file types (mcd, tiff, etc.) can produce mismatching channel lists.",
                      "json_requires_roi": "Please select an ROI before importing blend parameters from JSON.",
                      "custom_metadata_error": "Could not import custom metadata. Ensure that: \n \n- the dataset " \
                                        "containing the images is uploaded first" \
                                        "\n - a column titled `Channel Label` contains the desired labels to use for"
                                               "session naming"
                                               "\n - the number of rows matches the number of " \
                                        "channels in the current dataset. \n",
                      "metadata_format_error": "Warning: the edited metadata appears to be incorrectly formatted. " \
                                    "Ensure that the number of " \
                                    "channels matches the provided channel labels.",
                      "invalid_annotation_shapes": "There are annotation shapes in the current layout. \n" \
                                "Switch to zoom or pan before removing the annotation shapes.",
                      "invalid_dimensions": "The dimensions of the mask do not agree with the current ROI.",
                      "quantification_missing": "Quantification requires the following inputs: \n\n"
                                                "- an ROI with a compatible mask that has been applied to the canvas. \n"
                                                "- at least one channel/biomarker selected for quantification. \n\n"
                                                "Please review the required inputs.",
                      "possible-disk-storage-error": "The imported data could not be read/cached. \n"
                                                     "Check that there is sufficient disk storage to conduct analysis"
                                                     " (typically 2x the size of the imported files), and that the "
                                                     "imported files share the same panel.",
                      "lazy-load-error": "Error when loading data from the imported file. Check that the dataset "
                                         "delimiter does not have any overlapping characters with any of the filenames, "
                                         "or ROI names. ",
                      "invalid_query": "Error when generating ROI query. Ensure that: \n\n"
                                       "1. If querying a random subset, that images have been imported and "
                                       "the current canvas contains at least one marker. \n\n"
                                       "2. If querying from the quantification/UMAP tab: "
                                       "\n"
                                       "\t\n -quantification results are loaded"
                                       "\t\n -the corresponding images for quantified ROIs are loaded, and an image "
                                       "has been generated in the main canvas"
                                       "\t\n -ROI naming in the quantification sheet matches the ROI names in the session."}


class ToolTips(BaseModel):
    """
    Contains hash containing a tool tip hover text `tooltips` attribute

    :return: None
    """
    tooltips: dict = {"delimiter": "Set a custom delimiter for the string representation of datasets. "
                                "Should be used if imported datasets contain filenames or identifiers with"
                                "string overlap with the current delimiter.",
                      "import-tour": "Click here to get a tour of the components required for dataset import.",
                      "local-dialog": "Browse the local file system using a dialog. "
                                      "IMPORTANT: may not be compatible with the specific OS.",
                      "delete-selection": "Remove the current data collection. (IMPORTANT): cannot be undone.",
                      "roi-refresh": "Refresh the current dataset selection and canvas figure. "
                                    "Can be used if the ROI loading or canvas has become corrupted/malformed.",
                      "channel-mod": "Select a channel in the current blend to \nchange colour, "
                                    "pixel intensity, or apply a filter. Click for more information.",
                      "annot-reimport": "Re-import the current ROI annotations into the quantification "
                                            "results. Annotations must be re-added each time the quantification "
                                            "results are re-generated, or if annotations were generated "
                                            "without quantification results.",
                      "quantify-channels": "Warning: the time required for quantification will grow linearly as "
                                           "both the number of channels, and number of mask objects increases. "
                                           "Improve performance by reducing the number of channels quantified.",
                      "mask-name-autofill": "Matching the mask name to the corresponding ROI name will link the "
                                            "quantification results to region annotations and ROI gating.",
                      "max-viewport": "Modify the maximum viewport width for the main canvas. Use smaller values "
                                      "if the canvas image spills over into the side tabs, or larger values to "
                                      "permit the canvas to fit a larger proportion of the screen. The default is "
                                      "set at 150.",
                      "set-blends": "Save the current canvas channels in a named blend. Blends can be quickly toggled"
                                    " to compare marker expression across the current ROI.",
                      "cluster-proj": "IMPORTANT: Using circles for cluster projection for > 1000 mask objects will "
                                      "lead to slow performance."}


class PanelMismatchError(Exception):
    """
    Raise this exception when datasets with different panel lengths are uploaded into the same session
    """

class DataImportError(Exception):
    """
    Raise when imported data cannot be read fully into the session, likely due to a disk storage error
    """

class LazyLoadError(Exception):
    """
    Raise when the lazy loading feature of the `FileParser` class doesn't produce an ROI dictionary
    containing channel arrays
    """

def file_import_message(imported_files: list):
    """
    Generate the import alert for files read into the session
    """
    unique_suffixes = []
    message = "Read in the following files:\n"
    for upload in imported_files:
        suffix = pathlib.Path(upload).suffix
        message = message + f"{upload}\n"
        if suffix not in unique_suffixes:
            unique_suffixes.append(suffix)
    message = message + "\n Select a region (ROI) from the data collection dropdown to begin analysis."
    return message, unique_suffixes


def add_warning_to_error_config(error_config: Union[dict, None], alert: Union[str, None]) -> dict:
    """
    Parse an existing error/message/warning dictionary and add the current session warning
    """
    error_config = {"error": None} if (error_config is None or "error" not in error_config) else error_config
    error_config["error"] = alert if alert else ""
    return error_config
