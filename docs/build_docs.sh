#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
rm -r $SCRIPTPATH/build/
rm $SCRIPTPATH/source/rakaia*.rst
sphinx-apidoc -o $SCRIPTPATH/source/ $SCRIPTPATH/../rakaia/
make -C $SCRIPTPATH/ html
