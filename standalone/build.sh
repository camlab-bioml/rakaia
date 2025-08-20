#!/usr/bin/env bash

pyinstaller $1 --noconfirm --clean
chmod 777 dist/*
chmod +x dist/*
