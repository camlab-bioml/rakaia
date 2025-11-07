# -*- mode: python ; coding: utf-8 -*-

import distutils.util
import sys
import os
from rakaia._version import __version__
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

COMPILING_PLATFORM = distutils.util.get_platform()
safe_name = f"rakaia_{COMPILING_PLATFORM}_{__version__}".replace(".", "_").replace("-", "_")

with open('../requirements.txt') as f:
    required = f.read().splitlines()

keep_capital = ['Pillow', 'Cython', 'OpenSSL', 'Flask-HTTPAuth', 'PyWavelets']
required = [elem.split("==")[0].replace("-", "_") for elem in required]
required = [elem if elem in keep_capital else elem.lower() for elem in required]

additional_deps = []
for pkg in required:
    additional_deps += collect_data_files(pkg)
    additional_deps += copy_metadata(pkg)

additional_deps += collect_data_files("rasterio", include_py_files=True)

all_data = additional_deps + [
    ('../rakaia/templates', 'rakaia/templates'),
    ('../rakaia/static', 'rakaia/static'),
    ('../rakaia/assets', 'rakaia/assets')
]

icon_path = "../rakaia/assets/rakaia.ico"
if not os.path.isfile(icon_path):
    icon_path = None

block_cipher = None

a = Analysis(
    ['../rakaia/wsgi.py'],
    pathex=[],
    binaries=[],
    datas=all_data,
    hiddenimports=[
        'rasterio.sample',
        'rasterio.plot',
        'rasterio.warp',
        'rasterio.enums',
        'rasterio.vrt'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=safe_name,
    debug=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon_path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=safe_name + "_dist"
)
