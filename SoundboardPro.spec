# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os

# Use SPECPATH to get the directory containing this .spec file
# This makes the build portable across different machines
spec_root = os.path.abspath(SPECPATH)
src_path = os.path.join(spec_root, 'src')
web_path = os.path.join(src_path, 'web')
vbcable_path = os.path.join(spec_root, 'vbcable')
app_path = os.path.join(src_path, 'app.py')

datas = [(web_path, 'web'), (vbcable_path, 'vbcable')]
binaries = []
hiddenimports = ['eel', 'bottle', 'gevent', 'geventwebsocket', 'gevent.ssl', 'gevent._ssl3', 'pygame', 'pygame.mixer', 'pygame.sndarray', 'pygame.base', 'sounddevice', 'scipy', 'scipy.io', 'scipy.io.wavfile', 'numpy', 'keyboard', 'psutil', 'tkinter', 'tkinter.filedialog']
tmp_ret = collect_all('eel')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pygame')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sounddevice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [app_path],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'cv2', 'tkinter.test', 'unittest', 'pytest'],
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
