#!/bin/sh

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
find $(dirname "$SCRIPTPATH") -type d -name rakaia_cache -exec rm -rf {} \;
rakaia
