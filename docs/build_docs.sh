#!/usr/bin/env bash
rm -r build/
rm source/rakaia*.rst
sphinx-apidoc -o ./source ../rakaia/
make html
