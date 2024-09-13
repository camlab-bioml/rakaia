# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.20.0] - 2024-09-13

### Added

- plugin ecosystem for quantification results: leiden clustering, random forest (existing annotation, object mixing)
- Keyboard Event listener for mask toggle (up arrow)

### Changed

- grouped clustergram for categorical UMAP projections instead of heatmap

## [0.19.0] - 2024-08-29

### Added

- script to install using `pyapp` (UNIX systems)
- Cluster projection distribution table with selected cluster sub-categories
- Image prioritization using similarity scores computed from a cluster proportion

### Changed

- slight modification to `RegionAnnotation` pydantic implementation to be compatible with `pyapp`
- `glasbey` palettes and better categorical identification for UMAP overlay variables

### Fixed

- update from db or JSON for consecutive panel changes
- multi cluster color assignment import with multiple ROIs
- string casting for cluster category distributions
- UMAP figure patch now considers categorical overlays based on number of unique values
- quantification to cluster transfer for old pipeline (imcPQ) syntax
- proper selection of objects through UMAP categorical legend if initial zoom hasn't been used

## [0.18.0] - 2024-08-09

### Added

- Ability to transfer quantification columns to cluster projection per ROI
- Median and standard deviation to region summary #107
- Ability to toggle side placement of canvas tools sidebar #108

### Fixed

- Check for signal retention for channel gallery thumbnails after `Image.Resampling.LANCZOS` down-sampling
- Don't enable zoom on channel-wise gallery due to varying ROI dimensions
- Add panel length check in addition to key match for JSON, db config updates
- More robust checks for importing files when ROIs already exist in the current session

### Changed

- Recursively go through channel gallery children on label changes instead of re-rendering all components
- Do not update channel gallery on zoom automatically: use button trigger to update
- Canvas mode bar moved to center #108

## [0.17.0] - 2024-07-23

### Added

- sphinx docstrings for classes
- Dropdown to enable multiple cluster annotation import and toggle

### Changed

- Rename (rakaia) and code clean (pylint compatibility)
- modify cache clearing conditions on new app initialization
- Set gating parameter slider steps smaller by half (0.005)

### Fixed

- Proper mass deletion of annotations across multiple categories
- Make ROI querying from quantification work across multiple MCDs

## [0.16.0] - 2024-07-09

### Added

- Exception thrown when the file parser doesn't successfully parse any ROIs
- Named blend configurations can be saved and toggle using a dropdown menu,
to rapidly change the canvas among sets of blended channels
- Mask name to ROI matching from `steinbock` pipeline naming conventions
- conversion parser for intensity h5ad from `steinbock` to internal dataframe
- Mask input options now saved in JSON and database #104
- Animation to mantine color swatch buttons on click
- Ability to gate a custom list of mask IDs provided as a string input
- Set steinbock mask dtype os environment variable on app startup, related to
[this](https://github.com/BodenmillerGroup/steinbock/issues/132)

### Changed

- Major UI revamp prior to public release
- Improved shape region parsing for redrawn/edited shapes
- Better generalizations for conditions for tiff and HTML download
- Switch to `glasbey` for palette generation
- Switch to `steinbock` `scipy.ndimage` implementation for quantification and region props
- add local persistence to mask-related inputs
- Switch to `scipy.ngimage` to compute mask object overlap on drawn annotations
- Bump import max for `dash-upload` image components (up to 50gb)
- Channel gallery thumbnails now use smaller add icon without text
- Modify heatmap to show down-sample by default with reporting total object counts
- Run UMAP on current channels in heatmap
- refactor API module naming for wider release
- Use partial functions as switch statements for file reading in `RegionThumbnail`, `FileParser`
- dedicated category dropdown for click annotations
- JSON panel update now requires all keys to match, not just length
- Cap length of frequency distribution table in quant tab

### Fixed

- Prevent callback on gallery channel view when not enabled
- Handle overflow for long ROI names in ROI data table using scroll

### Removed

- `Undo last annotation` for regions (annotations table has selectable deletion)
- Preset hover (converted to static input + button)

## [0.15.0] - 2024-05-17

### Added

- warning message for ROI query when images or quant results are missing. #101
- CLI script for autoscaling ROIs from mcd
- Min amd max dimension limits (optional) + keyword searching for ROIs in gallery
query with improved recursive searching to match the query number
- Percentage distribution for UMAP categories

### Fixed

- missing callback output for opening in-browser quantification modal
- fully refresh channel selection on JSON/db update
- Proper color updates to the blend hash when using autofill #102
- Re-import exported custom metadata works as expected
- proper str casting for cluster annotations as circles
- Upper bounds of 0 from JSON style imports are no longer reset to the default percentile
- Change default CSV annotation value to `Unassigned` to avoid empty parsing by pandas on re-upload

### Changed

- ROI selection menu moved to main canvas to facilitate more consistent use. #97
- mask boundaries using `skimage` now use the innermost pixels as opposed to outer
- Better legend and scalebar x-axis range placement for varying ROI dimensions
- Specify Ubuntu version 22.04 in Docker
- UMAP channel overlay now uses `dash.Patch` for faster re-render (only numeric to numeric)
- Move `PanelMismatchError` import warning into a modal alert

## [0.14.0] - 2024-04-18

### Added

- Tab for marker correlation: set a target and baseline marker with masking to get
proportion of marker overlap between mask/image and target to baseline within mask objects + Pearson pixel correlation
(uses current channel blend parameters for filter and threshold) (compatible with zoom)
- Button to autofill single mask upload names with the current ROI identifier
- Mask object counter in the quantification modal when a mask is enabled
- Exception thrown when lazy loading doesn't work for the current ROI, possible due to the string delimiter
- Generalize the mongoDB database access by exposing the connection string as an input: Allows custom
configuration of mongoDB instances
- Option to select specific cluster categories for mask projection
- Checklist toggle to bulk annotate all current canvas shapes (without, defaults to most recent)

### Fixed
- Update canvas shape filtering on freeform draw to clear errors caused by
[plotly 4462](https://github.com/plotly/plotly.py/issues/4462)
- Fix switch trigger to disable ROI switching on keyboard
- Mask objects in ROI thumbnails are now all represented by the same intensity (255)
- Do not reset channels to quantify on ROI changes
- Proper ROI querying with tiff files from UMAP
- Proper blanking of the UMAP plot when the quantification results are updated in-browser
- Freeform rectangle coordinates now parsed from either left or right dragging
- Fixed bug where annotating from UMAP overwrites the annotation hash instead of the quantification sheet
- PDF writer now includes all annotations in loop

### Changed
- Enforce ROI change can occur only on the canvas tab
- Edit the uirevision on shape clearing to resolve [plotly 2741](https://github.com/plotly/dash/issues/2741):
modify the uirevision variable while maintaining truthy value
- Better visibility for channels set to white in the dash ag grid (label set to black)
- minimize canvas compression level to speed up image generation
- More lenient cluster sheet import (now only verifies required column names)
- Wrap gallery tiles in loading screen
- Do not read ROI image for adjusting canvas size/dimensions

## [0.13.0] - 2024-03-12

### Added

- Custom string delimiter input to change the string representation of ROIs within the session
- Data refresh button for refreshing ROI selection in case of a `PreventUpdate` exception in the canvas
callback
- Add poetry files for development option
- Custom exception for data reading error at likely callback point where disk storage errors occur
- Added integrated (total) signal to channel summary statistics
- single or multichannel gating from quantification results: gating can be applied to a canvas mask and
combined with clustering labels (mask and circle)
- Add annotation from gated cell list to quantification results or CSV object list output
- removing the most recent annotation will also apply to the quantification frame annotations: will
set any annotated objects in quantification to the default `None`
- Delete annotations by choice from the annotation preview table #89
- Toggle switch for enabling ROI change using left and right arrow keys
- Toggle button for select/deselect all for in-browser quantification
- Toggle button for disabling in-session messages (Default is to enable for safety)
- Button to re-import ROI annotations into the quantification frame when the quant frame is re-generated

### Changed

- Annotation dictionary entries internally represented by `pydantic` base model
- Use only unique aliases in canvas legend: allows for channels representing the same measurement
to be combined with the same colour and represented only once for simplicity
- JSON and mongoDB documents now contain cluster colour annotations if they exist on export
- JSON export includes aliases
- MCD import now includes the acquisition ID in the ROI name in addition to the acquisition description
to ensure unique ROI naming
- modify UI theme
- change `debug` command line option to `production-mode`: wraps debugging and production-level WSGI server
into one command for concurrent deployments
- image downloads will now include the ROI name in the file name output #92
- Annotation preview table now contains the `preview` column which describes either the bounds
of the annotation or the objects contained within in a short string


### Fixed

- Added additional checks for panel length mismatches when files are not imported at the same time
- Add ability to retain cluster assignments for individual ROIs when toggling
- Cluster annotations can now be added to the canvas legend via toggle: can replace the channel labels
when clusters are applied
- `readimc` updated to `0.7.0` to include reading imc files with `strict=False` for leniency on corrupted data
- Add CLI option for session cache storage with a default generated by `tempfile`
- fixed improper array index type for quantification in the browser
- fix attribute errors on HTML canvas export that block the download
- ROI selection deletion now reflected dataset preview table

## [0.12.0] - 2024-01-26

### Added

- mongoDB database persistence: Users can use a registered a mongoDB account for the `rakaia-db` Atlas instance
to and/remove current and saved session configurations
(blend parameters, global filters, channel selection, naming/aliases)
(Note: uses an 0 Sandbox version for initial testing)
- Custom exception class for panel length mismatches on file parsing: verifies panel lengths both within single uploads
and among all files in multi-upload
- Optional integer input to down-sample the heatmap for proper rendering
- Ability to export objects inside region annotations without having a matching quantification sheet:
object ids (such as cells) can be exported in CSV format if the annotation has a corresponding mask

### Changed

- toggleable child class of `dash_extensions` `Serverside`: ability to add string keys
to Serverside objects to allow overriding of previous disk caches to
reduce storage requirements and eliminate duplicate data writing for the same `dcc.Store` objects. Default is enabled,
can be turned off in CLI options. Enables concurrency in shared instances #85
- JSON I/O supports the values for global filters
- Apply global filter to ROI query thumbnails
- Modal alerts now use `pydantic` basemodel
- Session downloads are now removed from the collapsible and use individual `dcc.send_file`
prompts instead of an `html.A` hyperlink
- custom scalebar value now sets the length of the scalebar by the value input and combines
with the pixel ratio
- ROI thumbnail queries can now be generated from tiff and txt files
- Download for the UMAP projection moved into the UMAP configuration modal
- Annotation mask downloads (tiff in zip) will include the original mask name used to generate
the annotations, if provided #88
- Make the metadata import more generalizable based on columns present: look only for the label column
and length match
- Remove redundant cell outline computation on single mask import (boundary is computed by default)

### Fixed

- imported JSON configurations will now match the proper length of datasets
imported from h5py
- Annotating the UMAP from the UMAP plot will trigger the recent annotation to be applied to the visualization
- Fix improper range slider tick markers and maximum for channels with a max below 1
- Add global filter variables to annotation PDF writer
- Fixed storage step of underlying image in `CanvasImage` to allow export to tiff with mask or grid mods
- Fixed overwriting previous annotation shapes when annotation categories are changed: shape parsing
for annotations will now consider only the most recent shape drawn
- freeform drawn annotations will be filtered to include only indices within the boundary of the image #87
- Fixed incorrectly formatted function for importing custom metadata table (channel labels and names)
- Fixed proper setting of debugging mode for both Flask and dash server components. By default, debugging mode will be
enabled for troubleshooting, with the possibility of switching the default in subsequent versions
- Fixed improper shape clearing on canvas layout adjustments
- Fixed improper cluster imports: casting to str and fixing `skimage.measure` `regionprops` ordering for circles

## [0.11.0] - 2023-12-21

### Changed

- Added persistence variable logging to select variables (session + canvas
appearance) that are cached in the browser for future sessions based on user preferences
- Additive image blending for channels is now handled through `numexpr` instead of numpy
for faster performance (applies to additive blending, recolouring, and intensity thresholds)
- Modify pattern matching for mask names to ROI names in datasets and quantification sheets to include partial match of
mask to ROI
- Move the canvas tiff download out of the download dropdown to prevent using
Serverside storage for every canvas image (saves tmp disk space): now uses `dcc.send_file`
instead of an `html.A` hyperlink

### Added

- Toggle feature to cast the canvas legend as horizontal instead of vertical
  (horizontal = all channel names on one line)
- Added ability to add custom colour swatches through CLI as a comma separated string #81
- Added ability to upload cluster assignments in CSV format to annotate an applied mask
by colour (auto-generated or default random) #82
- Ability to re-import point annotations and populate shapes in canvas #83
- First draft: ability to read in quantification sheets from anndata/h5ad #54
- CLI toggle option to toggle the backend array store type from 32 byte float (default) to
unsigned 16 byte integer, with a tradeoff of precision vs. memory and speed

### Fixed

- Fixed extraneous callback on the canvas redraw the channel order is triggered
but the order has not changed
- Fixed improper intensity histogram max range due to sub-setting

## [0.10.0] - 2023-11-27

### Changed

- Internal API changed: new directory structure to accommodate
future data classes for the internal data stores
- quantification: the expression bar plot is now replaced
with a heatmap that changes dynamically on UMAP region selection.
Channels can be added/removed using a checklist
- The gaussian filter is now changed to `GaussianBlur` from the cv2
package.
- Loading screens for the data import and data switch are now toggle-able
with the CLI option `-dl` to disable load screens when importing or
changing to new data. Allows for smoother transitions between ROIs #73
- Next and previous ROI buttons are grayed out when the next or previous
ROIs are not available, respectively. #75
- mask object outlines are now computed with `skimage` to drastically
increase speed and avoid pixel iteration
- Imported imaging filenames are now sorted using natsort by default (feature can
be disabled in the session configuration modal)

### Added

- Added a sigma value input for the gaussian filter as above.
Default value is 1 with steps of 0.1. Used in the new gaussian
blur filter
- ROI queries from the quantification tab will apply the mask
and fill in the cells show in the UMAP with white
- Ability to auto assign colours from the swatches #72
- Added functionality for keyboard-directed switches between
ROIs using the left and right arrow keys #79

### Fixed

- Fixed channel filter values being reset to a default
when a channel does not have a filter. Now, the values
will remain from the last channel to get a filter applied
- Fix improper shapes clearing and malformed shapes
that cause erroneous "half-shapes" to appear on the canvas
- Check for very long ROI names or ones containing a space
that may interfere with proper separation in the dropdown menu. #71
- Fixed UMAP clearing when in-app quantification updates the
heatmap: UMAP plot will be hidden until new UMAP projections
are computed
- Fixed improper mask projection over canvas subset
when regions are exported to PDF
- Fixed improper conversion of cell mask to boundaries
when numpy floats are not used in the mask array

## [0.9.0] - 2023-11-07

### Changed

- Click annotations now have a circle surrounding the pixel by default
  (radius of 4 pixels) that is editable with adjustable radius size
- Channel image gallery is now clickable through pattern matching:
channels can be added to the canvas through a gallery click
- Reset the mask option to None on an ROI switch to avoid the
assumption that the current mask should be applied to a different image
- Cache the quantification measurements CSV and UMAP coordinates
as Serverside objects to speed up interactive UMAP for large datasets
- Annotations linked to quantification sheets are parsed to identify
which column should be used to match the annotation to the ROI
- Set the default drag mode to zoom on a canvas export to HTML
  (prevents spurious shapes from being drawn offline)
- Cast all return values for the quantification frame as Serverside
objects to speed up manual annotations for large data frames
- Current zoom is retained on ROI change if the dimensions of the new
ROI match exactly. Otherwise, a canvas refresh is made
- Annotation categories can now be created in both the canvas
side tab and the quantification option tab
- Improved heuristics for matching the quantification sheet
columns to the current ROI name using either description or sample. #69

### Added

- Dataset query: mcd files can now be queried across ROIs to get
clickable thumbnails for different ROIs that can then be loaded into
the canvas
- ROI name is now visible in the session outside the dropdown menu #68
- Button to reset current channel percentile scaling to the default
of the 99th percentile. #67
- Additional session configuration variables in top right corner:
can now switch on scroll zoom on the canvas (default is not enabled)
- Ability to overlay grid on canvas with white lines to assist in
cell counting
- Link between sub-setting cells on the UMAP and sending a query to
the query tab to display relevant ROIs by descending cell count
- Pattern matching on mask and ROI names to automatically find the
proper mask on an ROI change
- Users can now annotate quantification sheets from interactive
UMAP subsetting in addition to annotations on the canvas. Annotation
category options are the same as the options listed for canvas-derived
annotations

### Fixed

- Edits to the pixel histogram and range slider for images
with pixel ranges between 0 and 1
- Added additional logic to parse for ROI names in quantification
sheet #69
- Edited callback triggers for the expression bar plot to update
properly on a mode change when a subset is being used

## [0.8.0] - 2023-10-12

### Changed

- Region statistics table now contains separate rows for each
region drawn with a shape (rectangle or freeform). Previously, all metrics were averaged across
drawn regions
- Permit the canvas to be resized when zoom or drawn shapes are applied
- Allow bulk upload of masks, using default base names as the labels
when more than one mask is uploaded at once #59

### Added

- Ability to click a pixel on the canvas and annotate it
with a cell type and export point annotations in CSV format
- Ability to sort channels alphanumerically in the channel
selection dropdown (default is original channel order) #61
- Add port input as CLI option (default is 5000)
- Ability to export annotations as masks (one tiff per
annotation class/grouping)
- Input for setting ratio of pixel to distance #62
- Ability to invert annotations placement on x-axis
- Ability to subset ROI based on freeform draw on export #65
- Set range on pixel intensity slider to new maximum to
make it easier to adjust pixels for channels with a very large
intensity range #60
- Buttons for previous and next ROI. #63
- Output click point annotations in CSV format with cell ID (if a
mask is applied) and ROI ID #64

### Fixed

- Fixed error in gallery zoom when the zoom box was outside
of the subset dimensions
- Fixed histogram tick marks for input channels with a max
intensity below 3 (divisor for the tick marks)
- Fixed shape removal on new canvas modification when
layers are added
- Fix missing tick marks on range slider when switching to channels with
lower bounds or numbers not easily divisible by 3


## [0.7.0] - 2023-09-18

### Changed

- Ensure scalebar value is always centered over the scalebar line
- Changed nested hash tables for arrays and blend parameters to
top level hash tables for better hierarchical representation #34
- Remove any previous rakaia cache dirs in the tmpdir prior to execution
- Channel modification menu is automatically updated with the latest
channel in the blend when the layers are updated. If a layer is removed,
then the last channel in the queue is transferred to the mod menu #57

### Added

- CLI input option for local file dialog with wxPython #38
- Input to add multiple annotation quantification columns #54
- Frequency table for current UMAP selection #54
- First draft of output report for annotations #53 (static)
- Add version in header banner, right justified
- Individual toggle buttons for the scalebar and legend to be
toggle visible individually #55
- Toggle ability to read either local filepaths or read
all files in a directory matching the appropriate image file extensions
- Ability to export the panel blend parameters as JSON and re-import (drag and drop)
- New warning on file import if multiple file type extensions are detected
- Message after file import listing the file obtained

### Fixed

- Fixed improper canvas resizing on layer change
- Fixed improper update of intensity slider from h5 due to improper
string to int conversion
- Fixed error in writing Nonetype to h5 datasets
- Fixed proportion/aspect ratio of interactive canvas export to HTML

## [0.6.0] - 2023-08-21

### Changed

- Auto sizing of the canvas now happens automatically on an ROI dimension switch
instead of manually

### Added

- Ability to add the rakaia cell type annotation to a quantification data frame from a
region annotation: supports zoom, rectangles, and open form (non-convex) shapes #48
- Added slider to change the size of the annotations (scalebar value and legend text)
- Add toggle for custom hovertext over a mask, adding the mask/cell ID to the hover #51

### Fixed

- Edits to callbacks on channel switching to avoid running callbacks when the channel
blend parameters haven't changed (happens when switching between channels in the mod menu)
- Fix the centering of the scalebar value over the scalebar and edit default scalebar size and
ratio of scale value to legend size

### Changed

- More explicit warning on quantification upload if the dimensions are not compatible

## [0.5.0] - 2023-08-10

### Changed

- Major updates to the UI, particularly for the image annotation/blending tab:
multiple inputs converted to being held in bootstrap offcanvas tabs
(one offcanvas for inputs and one for advanced canvas settings)
- Edit the canvas div to center the canvas by default and on resizing
- Make the pixel histogram collapsible and by default have it hidden to speed up channel switching
- Add font awesome icons to various buttons and remove default styling

### Added

- Add default argparse argument to open the browser window automatically on CLI run
- Add colour box swatches from dash mantine colour picker for preferred colours

### Fixed

- Make sure the minimum of the default upper bound for pixel scaling by
percentile is at least 1 #49


## [0.4.0] - 2023-08-01

### Added

- ability to import multiple masks as single channel tiffs, either with or without cell boundaries with range slider for mask opacity
- ability to import a measurements CSV with defined columns for marker expression by segmented cell
- UMAP plotting for the marker measurements as above by cell
- toggle scalebar and legend in the canvas
- add button to remove an ROI from the session #41
- ability to custom sort the channels in a dash ag draggable grid #36


### Fixed

- ability to perform multiple rounds of dataset import from drag and drop #16
- retain the currently selected channel in the mod menu when the ROI is changed #43

## [0.3.0] - 2023-07-10

### Added

- first draft of zoomed in coordinate navigation #26
- add ability to add the pixel intensity values to the hover template #19
- percentile scaling #22 #45

### Fixed

- proper pixel blending for overlapping pixels
- better pixel scaling (linear from lower to upper bound)
- callback separation for speed improvements
- remove dcc.Stores from dcc.Loading by default to improve app smoothness

### Changed

- convert the pixel intensity changer to a range slider instead of box on the histogram

### Removed

- local dash diskcache background manager (conflict with pickle on certain Windows machines)

### Security

## [0.2.0] - 2023-06-19

### Added

- Lazy loading for mcd files using dataset preview by ROI
- Ability to read IMC datasets from multiple sources including single and multi-page TIFF, mcd, .txt file


### Fixed

- #18
- #21
- #23
- #25
- #30
- #31
- #32
- #28
- #20
- #24
- #37
- #40


### Changed

- Output from each session will be the currently loaded ROIs depending on the type of file input
- Modal warnings will appear for improper uploads on dataset and custom metadata

### Removed

### Security

## [0.1.0] - 2023-05-29

Initial beta/development release for the Jackson lab group
