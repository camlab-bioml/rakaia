#!/bin/sh
#!/bin/sh

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
APP_ROOT="$(dirname "$SCRIPTPATH")"
cat $APP_ROOT/requirements.txt | xargs poetry add $APP_ROOT
