import pathlib
from pydantic import BaseModel

class DataImportTour(BaseModel):
    """
    Contains the steps used for the dash tour component for data import assistance
    """
    steps: list = [{'selector': '[id="upload-image"]',
                    'content': "Upload your images (.mcd, .tiff, etc.) using drag and drop. Should"
                               " be used only for datasets < 2GB"},
                {'selector': '[id="read-filepath"]',
                'content': "For large datasets (> 2GB) or multiple files, copy and paste either a filepath or directory and "
                               "read files directly by selecting Import local"},
                {'selector': '[id="show-dataset-info"]',
                'content': 'View a list of imported datasets and regions of interest (ROIs)'},
                {'selector': '[id="data-collection"]',
                'content': 'Select an ROI from the dropdown menu to populate the image gallery'
                               ' and begin image analysis'},
                {'selector': '[id="annotation-canvas"]',
                    'content': 'Create a multiplexed image in the canvas by selecting\n'
                               ' channels/biomarkers from the Channel selection dropdown.'}]

class AlertMessage(BaseModel):
    """
    This class returns a simple string alert message display in the `error warning modal
    """
    warnings: dict = {"blend_json_success": "Blend parameters successfully updated from JSON.",
                      "invalid_filepath": "Invalid filepath provided. Please verify the following: \n\n" \
                                        "- That the file path provided is a valid local file \n" \
                                        "- If running using Docker or a web version, " \
                                        "local file paths will not be available.",
                      "invalid_directory": "Invalid directory provided. Please verify the following: \n\n" \
                                            "- That the directory provided exists in the local filesystem \n" \
                                            "- If running using Docker or a web version, " \
                                            "local directories will not be available.",
                      "multiple_filetypes": "Warning: Multiple different file types were detected on upload. " \
                                        "This may cause problems during analysis. For best performance, " \
                                        "it is recommended to analyze datasets all from the same file type extension " \
                                        "and ensure that all imported datasets share the same panel.\n\n",
                      "json_update_success": "Blend parameters successfully updated from JSON.",
                      "json_update_error": "Error: the blend parameters uploaded from JSON do not " \
                                        "match the current panel length. The update did not occur.",
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
                      "quantification_missing_mask": "Quantification requires an ROI with a compatible mask that has been applied to the" \
                                    " canvas. Please review the required inputs.",
                      "possible-disk-storage-error": "The imported data could not be read/cached. \n"
                                                     "Check that there is sufficient disk storage to conduct analysis"
                                                     " (typically 2x the size of the imported files)."}


class PanelMismatchError(Exception):
    """
    Raise this exception when datasets with different panel lengths are uploaded into the same session
    """
    pass

class DataImportError(Exception):
    """
    Raise when imported data cannot be read fully into the session, likely due to a disk storage error
    """
    pass

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