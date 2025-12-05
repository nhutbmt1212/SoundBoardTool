@echo off
echo ========================================
echo  Soundboard Pro - Build Standalone EXE
echo ========================================
echo.
echo This will create a TRUE standalone EXE with:
echo   - Python runtime bundled
echo   - All libraries bundled (pygame, pyaudio, numpy)
echo   - VB-Cable installer bundled
echo   - No installation needed!
echo.
echo Output: dist\SoundboardPro.exe (~40-50MB)
echo.
echo User Experience:
echo   1. Double-click SoundboardPro.exe
echo   2. First run: Install VB-Cable (optional)
echo   3. Use app - no setup needed!
echo.
pause

echo.
echo Installing build dependencies...
pip install pyinstaller pygame-ce pyaudio numpy

echo.
echo Building executable (this takes 2-3 minutes)...
echo Please wait...
echo.
python build_exe.py

echo.
echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo Output: dist\SoundboardPro.exe
echo.
echo Next steps:
echo   1. Test: dist\SoundboardPro.exe
echo   2. Distribute: Upload to GitHub Releases
echo.
pause
