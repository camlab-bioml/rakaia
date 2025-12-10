# Contributing

## Issues and Bug Reporting

If you encounter a problem with rakaia, briefly search the [existing issues](https://github.com/camlab-bioml/rakaia/issues) to see if it has already been reported. If not, create a new issue with the following details:
1. A clear, descriptive title.
2. The version of rakaia used (This can be found in
the top right corner of the application window or using `rakaia -v`).
3. The deployment type of rakaia used (i.e. install from source,
`pyinstaller` standalone build, deployment using `Docker`, etc.)
4. General steps to reproduce the issue.
5. Expected and actual behavior.
6. If possible, reference or include the dataset type(s) and filetype(s) being used.
7. Any relevant console logs, screenshots, or environment details.

## Contributing guidelines

Pull requests for new features, dependency updates, etc.
are welcome. Please review the general guidelines below.

### Testing

When possible, please aim to provide unit tests for new classes and functions.
Unit tests for the backend API typically cover at least 95% of the code,
and aim to test edge cases, particularly for parsers.

### Code complexity

We use `radon` to monitor code complexity, and aim to have
all functions receive a B or higher for code complexity (admittedly,
some of the earlier classes and functions violate this standard.)

### Dependencies

rakaia uses exact dependency versioning in order to
make installation highly reproducible across the major OS systems;
this practice is optima for non-computational users and assists with
generating standalone builds. If you are introducing a new dependency,
please use the newest possible version that is compatible with the current
dependencies, and freeze the exact version in the `pyproject.toml`.

#### Updating the `requirements.txt` file

The `pyproject.toml` is the primary source for dependency installation
for rakaia. However, we also update the `requirements.txt` file
for creating standalone builds with `pyinstaller`. If you introduce or update
a dependency, please also update the `requirements.txt` to match. A useful
tool for this conversion is [toml-to-requirements](https://pypi.org/project/toml-to-requirements/) v0.2.1:

```commandline
pip install toml-to-requirements==0.2.1
toml-to-req --toml-file pyproject.toml
```

### Pre commit

rakaia checks commits using [pre-commit](https://pre-commit.com/)
with pre commit hooks that verify formatting and Python-specific
packaging standards. Developers will need to install `pre-commit`
before committing to a development branch.

### Actions & CI/CD

We use GitHub Actions to automatically check installation
and unit tests using `pytest`. An excellent tool to verify
that the Actions pass before opening a pull request is [act](https://nektosact.com/).
