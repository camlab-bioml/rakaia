#!/usr/bin/env bash

pyinstaller $1 --noconfirm --clean
chmod 777 dist/*
chmod +x dist/*

chmod 777 dist/*_dist/*
chmod +x dist/*_dist/rakaia*
