from setuptools import setup, find_packages
import re
import os

def read_version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, "rakaia", "_version.py")
    with open(version_file, "r") as f:
        content = f.read()
    version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="rakaia",
    version=read_version(),
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
    description="Scalable spatial biology analysis in the browser.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["imaging cytometry single-cell browser multiplexed spatial"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    entry_points="""
    [console_scripts]
    {program} = rakaia.wsgi:main
    """.format(program="rakaia"),
    license="Unlicensed",
    install_requires=required,
    python_requires=">=3.10.0",
)
