# create the whl distribution first
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPTPATH/setup.py bdist_wheel --dist-dir $SCRIPTPATH/dist
VERSION=$( rakaia -v | cut -d "v" -f 2 )

export PYAPP_PROJECT_NAME="rakaia"
export PYAPP_PROJECT_DEPENDENCY_FILE="$SCRIPTPATH/requirements.txt"
export PYAPP_PROJECT_VERSION=$VERSIOn
export PYAPP_EXEC_SPEC="rakaia.wsgi:main"
export PYAPP_PROJECT_PATH="$SCRIPTPATH/dist/rakaia-$VERSION-py3-none-any.whl"
# export PYAPP_IS_GUI=1
export PYAPP_EXE_NAME="rakaia"

# assumes installation of pyapp in the same directory as the rakaia git source: https://ofek.dev/pyapp/latest/how-to/
cd ../pyapp-latest && cargo build --release && mv target/release/pyapp rakaia_bin_$VERSION && chmod 777 rakaia_bin_$VERSION
./rakaia_bin_$VERSION self remove
