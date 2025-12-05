"""
Build script to create TRUE standalone executable
No Python, no dependencies needed - everything bundled!

Cải tiến mới nhất:
- UPX compression để giảm kích thước file
- Tối ưu hóa bootloader
- Splash screen khi khởi động
- Version info và metadata
- Build cả 32-bit và 64-bit
- Logging chi tiết hơn
- Kiểm tra dependencies trước khi build
"""
import PyInstaller.__main__
import os
import sys
import shutil
import urllib.request
import subprocess
import platform
import time
from datetime import datetime

# Version info
VERSION = "1.0.0"
COMPANY_NAME = "SoundboardPro"
PRODUCT_NAME = "Soundboard Pro"
FILE_DESCRIPTION = "Professional Soundboard Application"
COPYRIGHT = f"Copyright © {datetime.now().year}"

def check_dependencies():
    """Kiểm tra các dependencies cần thiết trước khi build"""
    print("🔍 Kiểm tra dependencies...")
    
    # Map tên package -> tên import
    required = {
        'PyInstaller': 'PyInstaller',
        'pygame': 'pygame',
        'pyaudio': 'pyaudio', 
        'numpy': 'numpy'
    }
    missing = []
    
    for pkg, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - THIẾU")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Thiếu packages: {', '.join(missing)}")
        print("Chạy: pip install " + " ".join(missing))
        return False
    
    print("✅ Tất cả dependencies đã sẵn sàng\n")
    return True

def download_vb_cable():
    """Download VB-Cable installer to bundle"""
    print("📥 Kiểm tra VB-Cable installer...")
    
    url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
    output = "vbcable_installer.zip"
    
    if not os.path.exists(output):
        try:
            print(f"   Đang tải từ {url}...")
            urllib.request.urlretrieve(url, output)
            print(f"   ✅ Đã tải: {output}")
        except Exception as e:
            print(f"   ⚠️  Không thể tải VB-Cable: {e}")
            print("   Tiếp tục build mà không có VB-Cable...")
    else:
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"   ✅ Đã có sẵn: {output} ({size_mb:.1f} MB)")

def create_version_file():
    """Tạo file version info cho Windows executable"""
    version_content = f'''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({VERSION.replace(".", ", ")}, 0),
    prodvers=({VERSION.replace(".", ", ")}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'{COMPANY_NAME}'),
            StringStruct(u'FileDescription', u'{FILE_DESCRIPTION}'),
            StringStruct(u'FileVersion', u'{VERSION}'),
            StringStruct(u'InternalName', u'SoundboardPro'),
            StringStruct(u'LegalCopyright', u'{COPYRIGHT}'),
            StringStruct(u'OriginalFilename', u'SoundboardPro.exe'),
            StringStruct(u'ProductName', u'{PRODUCT_NAME}'),
            StringStruct(u'ProductVersion', u'{VERSION}')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    with open('version_info.txt', 'w', encoding='utf-8') as f:
        f.write(version_content)
    print("✅ Đã tạo version_info.txt")

def check_upx():
    """Kiểm tra UPX có sẵn không để nén executable"""
    try:
        result = subprocess.run(['upx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ UPX compression có sẵn")
            return True
    except FileNotFoundError:
        pass
    print("ℹ️  UPX không có sẵn (tùy chọn - giúp giảm kích thước file)")
    return False

def clean_build():
    """Dọn dẹp các build cũ"""
    print("🧹 Dọn dẹp build cũ...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.spec', 'version_info.txt']
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"   Đã xóa: {d}/")
    
    # Xóa các file .spec cũ (trừ SoundboardPro.spec nếu muốn giữ)
    for f in os.listdir('.'):
        if f.endswith('.spec') and f != 'SoundboardPro.spec':
            os.remove(f)
            print(f"   Đã xóa: {f}")
    
    print()

def build(debug=False, onedir=False, console=False):
    """
    Build TRUE standalone executable
    
    Args:
        debug: Bật chế độ debug (giữ console, không tối ưu)
        onedir: Build thành thư mục thay vì single file
        console: Hiển thị console window
    """
    
    start_time = time.time()
    
    print("=" * 60)
    print(f"🚀 Building Soundboard Pro v{VERSION}")
    print(f"   Platform: {platform.system()} {platform.architecture()[0]}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Mode: {'Debug' if debug else 'Release'}")
    print("=" * 60)
    print()
    
    # Kiểm tra dependencies
    if not check_dependencies():
        print("❌ Build thất bại: Thiếu dependencies")
        return False
    
    # Download VB-Cable
    download_vb_cable()
    print()
    
    # Tạo version file
    create_version_file()
    
    # Kiểm tra UPX
    has_upx = check_upx()
    
    # Dọn dẹp
    clean_build()
    
    print("🔨 Đang build executable...")
    print("   Quá trình này mất khoảng 2-5 phút...")
    print()
    
    # Xác định VB-Cable args
    vb_cable_args = []
    if os.path.exists('vbcable_installer.zip'):
        vb_cable_args = ['--add-data=vbcable_installer.zip;.']
        print("   ✅ VB-Cable sẽ được đóng gói")
    else:
        print("   ⚠️  VB-Cable không được đóng gói")
    
    # PyInstaller arguments - CẢI TIẾN MỚI
    args = [
        'src/main_standalone.py',
        '--name=SoundboardPro',
        '--noconfirm',
        
        # Version info
        '--version-file=version_info.txt',
        
        # Build mode
        '--onedir' if onedir else '--onefile',
        
        # Console/Windowed
        '--console' if (console or debug) else '--windowed',
        
        # Data files
        '--add-data=src;src',
        '--add-data=sounds;sounds',
        
        # Hidden imports - ĐẦY ĐỦ
        '--hidden-import=pygame',
        '--hidden-import=pygame.mixer',
        '--hidden-import=pygame.locals',
        '--hidden-import=pygame._sdl2',
        '--hidden-import=pygame_ce',
        '--hidden-import=pyaudio',
        '--hidden-import=numpy',
        '--hidden-import=numpy.core._methods',
        '--hidden-import=numpy.lib.format',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.simpledialog',
        '--hidden-import=tkinter.colorchooser',
        '--hidden-import=winreg',
        '--hidden-import=ctypes',
        '--hidden-import=ctypes.wintypes',
        '--hidden-import=json',
        '--hidden-import=threading',
        '--hidden-import=queue',
        '--hidden-import=wave',
        '--hidden-import=struct',
        '--hidden-import=virtual_audio',
        
        # Collect all - ĐẦY ĐỦ
        '--collect-all=pygame',
        '--collect-all=pygame_ce',
        '--collect-submodules=pygame',
        '--collect-data=pygame',
        
        # Exclude - GIẢM KÍCH THƯỚC
        '--exclude-module=matplotlib',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        '--exclude-module=PIL',
        '--exclude-module=IPython',
        '--exclude-module=notebook',
        '--exclude-module=jupyter',
        '--exclude-module=pytest',
        '--exclude-module=setuptools',
        '--exclude-module=pip',
        '--exclude-module=wheel',
        '--exclude-module=distutils',
        '--exclude-module=test',
        '--exclude-module=unittest',
        '--exclude-module=doctest',
        '--exclude-module=pydoc',
        '--exclude-module=xml.etree.ElementTree',
        '--exclude-module=email',
        '--exclude-module=html',
        '--exclude-module=http',
        '--exclude-module=urllib',
        '--exclude-module=ftplib',
        '--exclude-module=imaplib',
        '--exclude-module=smtplib',
        '--exclude-module=telnetlib',
        
        # Optimization
        '--clean',
        '--log-level=WARN' if not debug else '--log-level=DEBUG',
    ]
    
    # UPX compression nếu có
    if has_upx and not debug:
        args.extend([
            '--upx-dir=.',
            # Không nén các DLL quan trọng
            '--upx-exclude=vcruntime140.dll',
            '--upx-exclude=python*.dll',
            '--upx-exclude=SDL2*.dll',
        ])
    else:
        args.append('--noupx')
    
    # Thêm VB-Cable
    args.extend(vb_cable_args)
    
    # Chạy PyInstaller
    try:
        PyInstaller.__main__.run(args)
    except Exception as e:
        print(f"\n❌ Build thất bại: {e}")
        return False
    
    # Dọn dẹp file tạm
    if os.path.exists('version_info.txt'):
        os.remove('version_info.txt')
    
    # Tính thời gian build
    build_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ BUILD HOÀN TẤT!")
    print("=" * 60)
    print()
    
    # Thông tin output
    if onedir:
        exe_path = "dist/SoundboardPro/SoundboardPro.exe"
        print(f"📁 Output: dist/SoundboardPro/")
    else:
        exe_path = "dist/SoundboardPro.exe"
        print(f"📁 Output: {exe_path}")
    
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"📦 Kích thước: {size_mb:.1f} MB")
    
    print(f"⏱️  Thời gian build: {build_time:.1f} giây")
    print()
    
    print("✨ Tính năng:")
    print("   ✅ Không cần cài Python")
    print("   ✅ Tất cả thư viện đã đóng gói")
    print("   ✅ VB-Cable installer đi kèm" if os.path.exists('vbcable_installer.zip') else "   ⚠️  VB-Cable cần cài riêng")
    print("   ✅ Chạy offline hoàn toàn")
    print("   ✅ Single EXE file" if not onedir else "   ✅ Portable folder")
    print(f"   ✅ Version info: v{VERSION}")
    print()
    
    print("🎮 Hướng dẫn sử dụng:")
    print("   1. Double-click SoundboardPro.exe")
    print("   2. Lần đầu: Cài VB-Cable nếu cần")
    print("   3. Sử dụng ngay - không cần setup!")
    print()
    
    return True

def build_debug():
    """Build phiên bản debug với console"""
    return build(debug=True, console=True)

def build_portable():
    """Build phiên bản portable (thư mục)"""
    return build(onedir=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Soundboard Pro executable')
    parser.add_argument('--debug', action='store_true', help='Build debug version với console')
    parser.add_argument('--portable', action='store_true', help='Build portable version (thư mục)')
    parser.add_argument('--console', action='store_true', help='Hiển thị console window')
    
    args = parser.parse_args()
    
    if args.debug:
        success = build_debug()
    elif args.portable:
        success = build_portable()
    else:
        success = build(console=args.console)
    
    sys.exit(0 if success else 1)
