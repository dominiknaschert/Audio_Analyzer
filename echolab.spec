# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Echolab

Build with:
    pyinstaller echolab.spec

Result:
    Windows: dist/echolab/echolab.exe
    Mac: dist/echolab.app
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        'scipy.signal',
        'scipy.fft',
        'scipy.special',
        'numpy',
        'pyqtgraph',
        'soundfile',
        'sounddevice',
        'librosa',
        'audioread',
        # PySide6 plugins
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages
        'matplotlib',
        'tkinter',
        'PIL',
        'cv2',
        'torch',
        'tensorflow',
        # Exclude other Qt bindings (we use PySide6)
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific build configuration
if sys.platform == 'win32':
    # Windows: Create .exe with separate folder
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='echolab',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='echolab',
    )
else:
    # Mac (and Linux): Use onedir mode for .app bundles
    if sys.platform == 'darwin':
        # Mac: Create .app bundle with onedir mode
        exe = EXE(
            pyz,
            a.scripts,
            [],
            exclude_binaries=True,
            name='echolab',
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=True,
            console=False,
            disable_windowed_traceback=False,
            argv_emulation=False,
            target_arch=None,
            codesign_identity=None,
            entitlements_file=None,
            icon=None,
        )
        
        coll = COLLECT(
            exe,
            a.binaries,
            a.zipfiles,
            a.datas,
            strip=False,
            upx=True,
            upx_exclude=[],
            name='echolab',
        )
        
        app = BUNDLE(
            coll,
            name='echolab.app',
            icon=None,
            bundle_identifier='com.echolab.app',
        )
    else:
        # Linux: Create single-file executable
        exe = EXE(
            pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,
            [],
            name='echolab',
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=True,
            console=False,
            disable_windowed_traceback=False,
            argv_emulation=False,
            target_arch=None,
            codesign_identity=None,
            entitlements_file=None,
            icon=None,
        )

