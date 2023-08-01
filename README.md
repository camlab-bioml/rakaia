# ccramic

Cell-type Classification (using) Rapid Analysis (of) Multiplexed Imaging (mass) Cytometry using Flask and Dash.

ccramic provides in-browser interactive analysis of IMC imaging datasets.

## Installation

ccramic can be cloned and installed locally using access to the Github repository

```
git clone https://github.com/camlab-bioml/ccramic.git && cd ccramic
```

From there, the user may either install with or without a conda environment:

### Without conda (not recommended)

ccramic can be installed locally without an environment or container,
but this is not recommended for dependency management:

```
pip install .
```

### With conda

conda is the recommended installation manager for ccramic. To install conda locally,
visit [this link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and
select the relevant operating system.

Once conda is installed:

```
conda env create -f envs/environment.yml
conda activate ccramic
pip install .
```

The `requirements.txt` file can also be used to install the required dependencies via pip:

```
pip install -r requirements.txt
```

Note that with the conda installation as above, installing from the `requirements.txt` should not be necessary.

## Updating local installations

From source, ccramic can be updated locally using the following commands:

```
# navigate to the directory where you cloned ccramic from github
cd ccramic
git switch main
git pull --all
pip install .
```

## Running ccramic

After installation, ccramic can be run through conda:

```
conda activate ccramic
ccramic
```

The user should then navigate to `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/` to access ccramic.

## Basic authentication

ccramic uses basic authentication upon a new session. The credentials are as follows:

* username: ccramic_user
* password: ccramic-1

**Note** that the basic authentication credentials are likely to change with as development builds update.

## Docker

ccramic can be run using Docker with the following commands (requires an installation of Docker):


```
docker build -t ccramic envs/
docker run -d -p 5000:5000 ccramic:latest ccramic
```

Navigate to the local address `http://0.0.0.0:5000/`

## For developers

ccramic can be run in editable mode with either configuration shown below, which permits source code changes to be applied to the application on the fly:

```
pip install -e .
ccramic
```

Installing an editable version through pip is also required to run unit tests:

```
pytest --headless --cov ccramic

```

COnversely, without app installation:

```
python ccramic/app/wsgi.py
```
