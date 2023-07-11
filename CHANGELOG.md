# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
