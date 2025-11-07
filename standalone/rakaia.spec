# -*- mode: python ; coding: utf-8 -*-

from rakaia._version import __version__

import distutils.util

COMPILING_PLATFORM = distutils.util.get_platform()

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

with open('../requirements.txt') as f:
    required = f.read().splitlines()

keep_capital = ['Pillow', 'Cython', 'OpenSSL', 'Flask-HTTPAuth', 'PyWavelets']

required = [elem.split("==")[0].replace("-", "_") for elem in required]
required = [elem.lower() if elem not in keep_capital else elem for elem in required]


from PyInstaller.utils.hooks import collect_data_files, copy_metadata

additional_deps = []
for extra in required + ['rasterio']:
    additional_deps += collect_data_files(extra)
    additional_deps += copy_metadata(extra)

additional_deps += collect_data_files("rasterio", include_py_files=True)

block_cipher = None

all_data = additional_deps + [('../rakaia/templates', 'rakaia/templates'),
('../rakaia/static', 'rakaia/static'), ('../rakaia/assets', 'rakaia/assets')]


a = Analysis(
    ['../rakaia/wsgi.py'],
    pathex=[],
    binaries=[],
    datas=all_data,
    hiddenimports = [
    'rasterio.sample',
    'rasterio.plot',
    'rasterio.warp',
    'rasterio.enums',
    'rasterio.vrt'],
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
    a.datas,
    a.zipfiles,
    [],
    name=f'rakaia_{COMPILING_PLATFORM}_{__version__}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon = "../rakaia/assets/rakaia.ico"
)
