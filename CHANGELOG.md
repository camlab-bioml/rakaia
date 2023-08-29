# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0]

### Changed

- Ensure scalebar value is always centered over the scalebar line

### Added

- CLI input option for local file dialog with wxPython #38
- Input to add multiple annotation quantification columns #54
- Frequency table for current UMAP selection #54
- First draft of output report for annotations #53 (static)
- Add version in header banner, right justified

### Fixed

- Fixed improper canvas resizing on layer change

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
