from setuptools import setup, find_packages

setup(
    name="ccramic",
    version="0.1.0",
    url='https://github.com/camlab-bioml/ccramic',
    project_urls={
        "Issues": "https://github.com/camlab-bioml/ccramic/issues",
        "Source": "https://github.com/camlab-bioml/ccramic",
    },
    author="Matthew Watson",
    author_email="mwatson@lunenfeld.ca",
    packages=find_packages(),
    package_dir={"ccramic": "ccramic"},
    package_data={'': ['*.json']},
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
    {program} = ccramic.main:main
    """.format(program="ccramic"),
    license="Unlicensed",
    install_requires=["pillow", "pandas", "matplotlib", "fpdf",
                      "pytest", "freeport", "numpy", "scikit-image", "anndata", "scanpy",
                      "phenograph", "seaborn", "httpx-oauth", "plotly", "napari",
                      "opencv-python-headless", 'scanpy', "anndata", "dash", "Flask>=2.2.2", "jinja2",
                      'dash-uploader==0.7.0a1', "dash-canvas", "packaging==21.3.0", "Flask-Caching",
                      "multiprocess", "dash-extensions", "dash_daq", "dash-google-auth", "dash-bootstrap-components",
                      "pyvips", "imctools", "tifffile", "dash[testing]", "selenium", "diskcache", "h5py",
                      "orjson", "dash>=2.9.0", "readimc", "Cython", "dash-auth", "Flask-HTTPAuth",
                      "kaleido"],
    python_requires=">=3.9.0",
)
