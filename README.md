# rakaia

rakaia: Scalable multiplex imaging dataset analysis in the browser

<p align="center">
    <img src="man/assets/app-preview.png">
</p>


rakaia provides streamlined in-browser analysis of multiplexed imaging datasets.
The rakaia viewer is capable of rapid, interactive analysis of large regions of interest (ROI)
from imaging technologies such as imaging mass cytometry (IMC),
Immunofluorescence (IF) and others. Tools in the rakaia analysis
suite include:

- pixel level analysis for publication-quality blended images
- object/segmentation detection
- region/focal annotation
- object quantification
- cluster and heatmap visualization
- dataset-wide profiling and multi-ROI search
- database support (mongoDB)

rakaia benefits from on-demand data loading and requires minimal upfront data
configuration for ease-of-use image analysis. It places no restrictions on
data or project size, permitting users to visualize and analyze hundreds of
regions or images in a single session.

Importantly, rakaia does not require any coding/scripting, or
any pre-defined project directories with specific file structures.


## Installation

rakaia can be cloned and installed locally using access to the Github repository

```
git clone https://github.com/camlab-bioml/rakaia.git && cd rakaia
```

From there, the user may either install with or without a conda environment:

### Without conda (not recommended)

rakaia can be installed locally without an environment or container,
but this is not recommended for dependency management:

```
pip install -r requirements.txt
pip install .
```

### With conda

conda is the recommended installation manager for rakaia. To install conda locally,
visit [this link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and
select the relevant operating system.

Once conda is installed:

```
conda create --name rakaia python=3.9
conda activate rakaia
# cd rakaia
pip install -r requirements.txt
pip install .
```

### with Make

rakaia contains a Makefile that wraps the pip installation
commands above. Installation can be done as follows:

```
# cd rakaia
make
make install
```

## Updating local installations

From source, rakaia can be updated locally using the following commands:

```
# navigate to the directory where you cloned rakaia from github
cd rakaia
# activate your env first, if using conda
# conda activate rakaia
git switch main
git pull --all
pip install -r requirements.txt
pip install .
```

## Running rakaia

After installation, rakaia can be run through conda or simply executed using the `rakaia` command:

```
conda activate rakaia
rakaia
```
The user should then navigate to `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/` to access rakaia.

### Help

The ClI options for running rakaia can be viewed using:

```
rakaia -h
```

Additional information on the CLI options available for running custom rakaia sessions can be
found in the documentation: https://camlab-bioml.github.io/rakaia-doc/docs/cli


The current version of rakaia can also be checked on the command line with the following (v0.4.0 or later):

```
rakaia -v
```

## Documentation

The official user guide documentation for rakaia can be
found [here](https://camlab-bioml.github.io/rakaia-doc/docs/installation)

## mongoDB

From rakaia v0.12.0, users can use a registered mongoDB account for the
`rakaia-db` mongoDB instance to import, save, and remove past saved configurations.
Please reach out to mwatson@lunenfeld.ca to request access to existing databases or to receive
information on configuring custom database instances.

## For developers

rakaia can be run in editable mode with either configuration shown below,
which permits source code changes to be applied to the application on the fly:

```
pip install -e .
rakaia
```

Installing an editable version through pip is also required to run unit tests:

```
pytest --headless --cov rakaia
```

Conversely, without app installation:

```
python rakaia/wsgi.py
```

By default, rakaia will run in debug mode from the command line, which
will apply source code changes on the fly. To disable this feature
of to use a production-level server from waitress, enable production mode:

```commandline
rakaia -pr
```

### Binary distribution

rakaia can be compiled and distributed as a binary using `pyapp`. The instructions for building
rakaia standalone binaries can be found under the [building section](BUILDING.md)

## Troubleshooting

Troubleshooting tips can be found in the `Troubleshooting`
section of the [documentation](
https://camlab-bioml.github.io/rakaia-doc/docs/troubleshooting)
