#!/bin/sh

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
APP_ROOT="$(dirname "$SCRIPTPATH")"
echo $SCRIPTPATH
pip install -r $APP_ROOT/requirements.txt
pip install $APP_ROOT
