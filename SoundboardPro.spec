# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('src', 'src'), ('sounds', 'sounds'), ('vbcable_installer.zip', '.')]
binaries = []
hiddenimports = ['pygame', 'pygame.mixer', 'pygame.locals', 'pygame._sdl2', 'pygame_ce', 'pyaudio', 'numpy', 'numpy.core._methods', 'numpy.lib.format', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.simpledialog', 'tkinter.colorchooser', 'winreg', 'ctypes', 'ctypes.wintypes', 'json', 'threading', 'queue', 'wave', 'struct', 'virtual_audio']
datas += collect_data_files('pygame')
hiddenimports += collect_submodules('pygame')
tmp_ret = collect_all('pygame')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pygame_ce')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src\\main_standalone.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'PIL', 'IPython', 'notebook', 'jupyter', 'pytest', 'setuptools', 'pip', 'wheel', 'distutils', 'test', 'unittest', 'doctest', 'pydoc', 'xml.etree.ElementTree', 'email', 'html', 'http', 'urllib', 'ftplib', 'imaplib', 'smtplib', 'telnetlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SoundboardPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
)
