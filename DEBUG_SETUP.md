# Debug Setup Issues

## Problem Analysis

### Issue 1: setup.bat nháy nháy tự tắt
**Cause**: Script exits too quickly without showing output

**Fix Applied**:
1. Added `pause` at end of root setup.bat
2. Better error handling in scripts/setup.bat
3. Clear messages at each step

### Issue 2: run.bat says Python not installed
**Cause**: Python installed but not in PATH for current session

**Fix Applied**:
1. After Python install, script tells user to close window
2. User must open NEW command prompt
3. Python will be available in new session

## Current Flow

### setup.bat (Root)
```batch
@echo off
cd /d "%~dp0"
call scripts\setup.bat
pause  ← Added this to prevent auto-close
```

### scripts/setup.bat
```batch
1. Check if Python exists
   - If yes → goto install_deps
   - If no → call check_python.bat

2. After Python install:
   - Show message: "Close window and run again"
   - pause
   - exit

3. install_deps:
   - Install Python packages
   - Install VB-Cable
   - Show completion message
   - pause
```

### scripts/check_python.bat
```batch
1. Check Python
   - If exists → exit /b 0 (success)
   - If not → download and install

2. After install:
   - exit /b 2 (special code = just installed)
```

## Expected Behavior

### First Run (No Python)
```
User: Double-click setup.bat

Output:
========================================
 Soundboard Pro - Auto Setup
========================================

[Step 1/3] Checking Python installation...

[!] Python not found - installing automatically...

[1/3] Downloading Python installer...
[OK] Download complete

[2/3] Installing Python (this may take 2-3 minutes)...
[*] Installing silently with PATH configuration...
[OK] Installation complete

[3/3] Cleaning up...
[OK] Cleanup complete

========================================
[!] Python Installation Complete
========================================

IMPORTANT: Python was just installed.
You need to:
  1. Close this window
  2. Open a NEW command prompt
  3. Run setup.bat again

Press any key to exit...
```

### Second Run (Python Installed)
```
User: Open NEW command prompt, run setup.bat

Output:
========================================
 Soundboard Pro - Auto Setup
========================================

[Step 1/3] Checking Python installation...

[OK] Python is already installed
Python 3.11.7

========================================
[Step 2/3] Installing Dependencies
========================================

This will install:
  - Python packages (pygame, pyaudio, numpy)
  - VB-Audio Virtual Cable (for Discord/Game routing)

[*] Starting installation...

(Installation proceeds...)

========================================
[Step 3/3] Setup Complete!
========================================

Next steps:
  1. Restart your computer (REQUIRED for VB-Cable)
  2. Run: run.bat
  3. Click "Audio Setup" to configure

Press any key to exit...
```

### run.bat
```
User: Double-click run.bat

Output:
========================================
 Soundboard Pro
========================================

Checking Python...
[OK] Python found
Python 3.11.7

Starting Soundboard Pro...

(App starts)
```

## Testing Steps

1. **Test setup.bat**
   ```
   - Double-click setup.bat
   - Should NOT close immediately
   - Should show output
   - Should pause at end
   ```

2. **Test Python install**
   ```
   - If no Python: Should install automatically
   - Should tell user to close and rerun
   - Should pause before exit
   ```

3. **Test run.bat**
   ```
   - Should check Python
   - Should show clear error if no Python
   - Should start app if Python exists
   ```

## Common Issues

### "Window closes immediately"
**Solution**: Added `pause` at end of setup.bat

### "Python not found after install"
**Solution**: Must close ALL command prompts and open NEW one

### "Can't see what's happening"
**Solution**: Added clear output messages and pause commands

## Files Modified

- ✅ `setup.bat` - Added pause at end
- ✅ `scripts/setup.bat` - Better flow and error handling
- ✅ `scripts/check_python.bat` - Returns special exit code
- ✅ `scripts/run.bat` - Better Python check messages
- ✅ `run.bat` - Calls scripts/run.bat

## Verification

Run these commands to verify:

```batch
REM Test 1: Check setup.bat doesn't auto-close
setup.bat
REM Should pause at end

REM Test 2: Check run.bat shows clear error
run.bat
REM Should show "Python Not Found" if no Python

REM Test 3: After Python install
REM Close window, open NEW command prompt
setup.bat
REM Should continue to dependency install
```

---

**Status**: Fixed
**Version**: 1.0.2
**Date**: December 5, 2025
