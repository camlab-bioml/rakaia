# ccramic

Cell-type Classification (using) Rapid Analysis (of) Multiplexed Imaging (mass) Cytometry using Flask and Dash.

## Installation

ccramic can be cloned and installed locally using access to the Github repository

````
git clone https://github.com/camlab-bioml/ccramic.git && cd ccramic
```

From there, the user may either install with or without a conda environment:

### Without conda (not recommended)

```
pip install . 
```

### With conda

```
conda env create -f environment.yml
conda activate ccramic
pip install . 
```

The `requirements.txt` file can also be used to install the required dependencies via pip: 

```
pip install -r requirements.txt
```

## Running ccramic

 After installation, ccramic can be run through conda: 

 ```
 conda activate ccramic
 ccramic
 ```

 The user should then navigate to `http://127.0.0.1:5000/` to access ccramic.

## Basic authentication

ccramic uses basic authentication upon a new session. The credentials are as follows:

username: ccramic_user
password: ccramic-1

**Note** that the basic authentication credentials are likely to change with as development builds update. 

