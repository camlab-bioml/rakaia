from setuptools import setup, find_packages
from rakaia.entrypoint import __version__, _program

setup(
    name=_program,
    version=__version__,
    url='https://github.com/camlab-bioml/rakaia',
    project_urls={
        "Issues": "https://github.com/camlab-bioml/rakaia/issues",
        "Source": "https://github.com/camlab-bioml/rakaia",
    },
    author="Matthew Watson",
    author_email="mwatson@lunenfeld.ca",
    packages=find_packages(),
    package_dir={"rakaia": "rakaia"},
    package_data={'': ['*.json', "*.html", "*.css", "*.ico", "*.js", "*.png", "*.yaml", "*.jpg"]},
    include_package_data=True,
    description="Large scale multiplex image analysis in the browser",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["imaging cytometry classifier single-cell"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9"
    ],
    entry_points="""
    [console_scripts]
    {program} = rakaia.wsgi:main
    """.format(program="rakaia"),
    license="Unlicensed",
    install_requires=["pillow", "pandas", "matplotlib",
                      "pytest", "freeport", "numpy", "scikit-image", "anndata", "scanpy",
                      "phenograph", "seaborn", "plotly",
                      "opencv-python-headless", 'scanpy', "Flask>=2.2.2", "jinja2",
                      'dash-uploader==0.7.0a1', "dash-canvas", "packaging==21.3.0",
                      "multiprocess", "dash-extensions==1.0.1", "dash_daq", "dash-google-auth",
                      "dash-bootstrap-components", "imctools", "tifffile", "dash[testing]", "selenium",
                      "diskcache", "h5py", "dash>=2.9.0", "readimc>=0.7.0", "Cython", "dash-auth", "Flask-HTTPAuth",
                      "scipy==1.10.1", "dash-draggable", "pytest-cov", "sd-material-ui", "dash-ag-grid",
                      "dash-mantine-components==0.12.1", "numexpr", "pymongo", "pydantic==1.10.8", "dash-tour-component",
                      "shortuuid", "mongomock", "steinbock", "glasbey", "dash-bio==1.0.2", "statsmodels==0.14.0"],
    python_requires=">=3.9.0",
)
