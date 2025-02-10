from setuptools import setup, find_packages
from rakaia import __version__, _program

with open('requirements.txt') as f:
    required = f.read().splitlines()

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
    description="Scalable multiplexed dataset analysis in the browser",
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
    install_requires=required,
    python_requires=">=3.9.0",
)
