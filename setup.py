from setuptools import setup, find_packages
from ccramic.app.entrypoint import __version__, _program

setup(
    name=_program,
    version=__version__,
    url='https://github.com/camlab-bioml/ccramic',
    project_urls={
        "Issues": "https://github.com/camlab-bioml/ccramic/issues",
        "Source": "https://github.com/camlab-bioml/ccramic",
    },
    author="Matthew Watson",
    author_email="mwatson@lunenfeld.ca",
    packages=find_packages(),
    package_dir={"ccramic": "ccramic"},
    package_data={'': ['*.json', "*.html", "*.css"]},
    include_package_data=True,
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["imaging cytometry classifier single-cell"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9"
    ],
    entry_points="""
    [console_scripts]
    {program} = ccramic.app.wsgi:main
    """.format(program="ccramic"),
    license="Unlicensed",
    install_requires=["pillow", "pandas", "matplotlib",
                      "pytest", "freeport", "numpy", "scikit-image", "anndata", "scanpy",
                      "phenograph", "seaborn", "plotly",
                      "opencv-python-headless", 'scanpy', "Flask>=2.2.2", "jinja2",
                      'dash-uploader==0.7.0a1', "dash-canvas", "packaging==21.3.0",
                      "multiprocess", "dash-extensions==1.0.1", "dash_daq", "dash-google-auth",
                      "dash-bootstrap-components", "imctools", "tifffile", "dash[testing]", "selenium",
                      "diskcache", "h5py", "dash>=2.9.0", "readimc", "Cython", "dash-auth", "Flask-HTTPAuth",
                      "scipy", "dash-draggable", "pytest-cov", "sd-material-ui", "dash-ag-grid"],
    python_requires=">=3.9.0",
)
