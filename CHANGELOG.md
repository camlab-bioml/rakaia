# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2023-11-02

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

### Added

- Dataset query: mcd files can now be queried across ROIs to get
clickable thumbnails for different ROIs that can then be loaded into
the canvas
- ROI name is now visible in the session outside the dropdown menu
- Additional session configuration variables in top right corner:
can now switch on scroll zoom on the canvas (default is not enabled)
- Ability to overlay grid on canvas with white lines to assist in
cell counting
- Link between sub-setting cells on the UMAP and sending a query to
the query tab to display relevant ROIs by descending cell count
- Pattern matching on mask and ROI names to automatically find the
proper mask on an ROI change

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
- Remove any previous ccramic cache dirs in the tmpdir prior to execution
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

- Ability to add the ccramic cell type annotation to a quantification data frame from a
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
