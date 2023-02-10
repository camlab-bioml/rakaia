from setuptools import setup, find_packages

setup(
    name="ccramic",
    version="0.1.0",
    url='https://github.com/matt-sd-watson/ccramic/',
    project_urls={
        "Issues": "https://github.com/matt-sd-watson/ccramic/issues",
        "Source": "https://github.com/matt-sd-watson/ccramic",
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
    keywords=["single-cell download repository"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9"
    ],
    entry_points="""
    [console_scripts]
    {program} = ccramic.main:run_app
    """.format(program="ccramic"),
    license="Unlicensed",
    install_requires=["streamlit", "pillow", "pandas", "matplotlib", "fpdf", "streamlit-aggrid",
                      "pytest", "freeport"],
    python_requires=">=3.9.0",
)