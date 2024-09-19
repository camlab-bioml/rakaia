# Building rakaia binaries

rakaia supports building binaries distributions through [pyapp](https://github.com/ofek/pyapp)
Binaries can be built for all major OS distributions (Windows, MacOS, and Linux).

## Downloading and installing the pyapp source

Before installing pyapp, ensure that rust is installed from [here](https://www.rust-lang.org/tools/install). The
website should be able to detect the rust distribution depending on the user's OS.

To build the rakaia binary, the source code for pyapp must bw downloaded and installed
in the same parent directory as the rakaia source. Follow the instructions [here](https://ofek.dev/pyapp/latest/how-to/)
for downloading the pyapp source. The parent directory structure should look like this:

```commandline
pyapp-latest/
    docs/
    scripts/
    src/
rakaia/
    benchmarks/
    cli/
    docs/
    envs/
    man/
    rakaia/
    scripts/
    tests/
```

Next, ensure that rakaia has been installed through source.
Follow the instructions [here](README.md) under `Installation`.

## Creating an OS-specific binary

Each binary distribution must be created on the target OS. For either MacOS
or Linux, navigate to the rakaia source directory, then run:

```commandline
source pyapp_unix.sh
```

For Windows, the commands should be run in a shell supporting bash commands
such as Git bash for Windows) using the corresponding shell script:

```commandline
source pyapp_windows.sh
```

pyapp will then use Python v3.10 from the [python standalone build](https://github.com/indygreg/python-build-standalone)
project as the interpreter and to manage dependencies. For installation target will be different dependin
on the OS:

- on Linux, installation will be under the .local/share/pyapp/rakaia directory
for the specific user
- on MacOS,
- on Windows, installation will be under the AppData/pyapp directory for the specific user

The final binary executable will be found in the pyapp directory, named as `rakaia_version` for either Linux or
macOS, and `rakaia_version.exe` on Windows.

## Executing

the rakaia executable requires the use of a terminal window to communicate with the underlying server.

On Windows, the executable can simply be opened by double-clicking the application icon. In some cases, it may require
admin permissions. For this, right-click the application and select `Run as administrator`

For MacOS, right-click the executable file and select `Open with` -> `Other`. Select `Enable` -> `All Applications`,
then navigate to `Applications` -> `Utilities` -> `Terminal`. Run the application by double clicking.

Opening the application should trigger a new Terminal window. On first run, it may take a while as the application creates the
necessary virtual environment.
