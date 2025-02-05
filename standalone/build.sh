#!/usr/bin/env bash

pyinstaller ./rakaia.spec --noconfirm --clean
chmod 777 dist/*
chmod +x dist/*
