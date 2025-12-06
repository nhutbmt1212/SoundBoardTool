"""Build standalone executable"""
import PyInstaller.__main__
import os
import shutil

# Paths
SRC = 'src'
WEB = os.path.join(SRC, 'web')
SOUNDS = 'sounds'
DIST = 'dist'
BUILD = 'build'

# Clean
for d in [DIST, BUILD]:
    if os.path.exists(d):
        shutil.rmtree(d)

# Build
PyInstaller.__main__.run([
    os.path.join(SRC, 'app.py'),
    '--name=SoundboardPro',
    '--onefile',
    '--windowed',
    '--icon=NONE',
    f'--add-data={WEB};web',
    '--hidden-import=pygame',
    '--hidden-import=sounddevice',
    '--hidden-import=scipy',
    '--hidden-import=numpy',
    '--clean',
])

# Copy sounds folder
dist_sounds = os.path.join(DIST, SOUNDS)
if os.path.exists(SOUNDS):
    shutil.copytree(SOUNDS, dist_sounds, dirs_exist_ok=True)
else:
    os.makedirs(dist_sounds, exist_ok=True)

print(f"\n✅ Build complete: {DIST}/SoundboardPro.exe")
