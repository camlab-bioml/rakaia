# Building rakaia standalone

rakaia supports building binaries distributions through [pyinstaller](https://pyinstaller.org/en/stable/).
Binaries can be built for all major OS distributions (Windows, MacOS, and Linux).


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
source build.sh
```

For Windows, the commands should be run in a shell supporting bash commands
such as `Git Bash for Windows` using the corresponding shell script above.

The resulting application bundle file can be found in the `standalone/dist` directory as either a file executable
without a file type extension (Linux), or as an exe file on Windows.

**For distributing the file, the bundle should be zipped prior to sharing so that file permission may be maintained
across machines/environments.**


## Executing

The rakaia executable requires the use of a terminal window to communicate with the underlying server.

On Windows, the executable can simply be opened by double-clicking the application icon. In some cases, it may require
admin permissions. For this, right-click the application and select `Run as administrator`

For MacOS, right-click the executable file and select `Open with` -> `Other`. Select `Enable` -> `All Applications`,
then navigate to `Applications` -> `Utilities` -> `Terminal`. Run the application by double clicking.

Opening the application should trigger a new Terminal window. On first run, it may take a while as the application parses through all dependencies and verifies the bundled interpreter.

## Troubleshooting

### Windows SDK dependencies

In certain cases, Windows user may need to install the [Windows Development SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/) to have all required dll libraries for rakaia on windows.
