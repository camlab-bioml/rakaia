from pydantic import BaseModel

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
                      "custom_metadata_error": "Could not import custom metadata. Ensure that: \n \n- The dataset " \
                                        "containing the images is " \
                                        "uploaded first" \
                                        "\n - the columns `Channel Name` and " \
                                        "`Channel Label` are present \n - the number of rows matches the number of " \
                                        "channels in the current dataset. \n",
                      "metadata_format_error": "Warning: the edited metadata appears to be incorrectly formatted. " \
                                    "Ensure that the number of " \
                                    "channels matches the provided channel labels.",
                      "invalid_annotation_shapes": "There are annotation shapes in the current layout. \n" \
                                "Switch to zoom or pan before removing the annotation shapes.",
                      "invalid_dimensions": "The dimensions of the mask do not agree with the current ROI.",
                      "quantification_missing_mask": "Quantification requires an ROI with a compatible mask that has been applied to the" \
                                    " canvas. Please review the required inputs."}


class PanelMismatchError(Exception):
    pass
