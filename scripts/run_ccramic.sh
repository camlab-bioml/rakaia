#!/usr/bin/env bash

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
find $(dirname "$SCRIPTPATH") -type d -name ccramic_cache -exec rm -rf {} \;
ccramic
