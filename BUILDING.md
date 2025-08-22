# Building rakaia standalone

rakaia supports building binaries distributions through [pyinstaller](https://pyinstaller.org/en/stable/).
Binaries can be built for all major OS distributions (Windows, macOS, and Linux).


## pyinstaller installation

Pyinstaller can be installed from pip similar to the package dependencies. Building a standalone requires that the rakaia
source code be cloned and the application installed from source in the current environment. The following pyinstaller version is compatible
with the rakaia dependency list:

```
pip install pyinstaller==6.11.1
pip install .
```

## Creating an OS-specific binary

Each standalone distribution must be created on the target OS. The following commands work across platform
from the rakaia home source directory:

```commandline
cd standalone/
source build.sh rakaia.spec
```

The spec file above compiles a single file executable, with compressed dependencies.
For a directory of uncompressed dependencies, switch the spec file:

```commandline
source build.sh rakaia_onedir.spec
```

### Building on Windows

A few additional considerations need to be made when building on Windows:

- the commands should be run in a shell supporting bash commands
such as `Git Bash for Windows`
- the minimal Python version for building should be `3.11` or greater to avoid
build errors with `rasterio`
- rakaia should not be built in a virtual env or conda

The resulting application bundle file can be found in the `standalone/dist` directory as either a file executable
without a file type extension (Linux), or as an exe file on Windows if single file is used.
For the onedir spec, a subdirectory containing a file executable and a `_internal` dependency
directory will be generated.

**For distributing the file, the bundle should be zipped prior to sharing so that file permission may be maintained
across machines/environments.**

## Executing

Please refer to the **INSTRUCTIONS.Rmd** file for
information on executing the bundles.
