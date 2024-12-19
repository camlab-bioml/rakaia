#!/usr/bin/env bash

pyinstaller ./rakaia.spec --noconfirm
chmod 777 dist/*
chmod +x dist/*
