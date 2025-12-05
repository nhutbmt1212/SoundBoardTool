# Hướng Dẫn Cài Đặt Tự Động 100%

## Cài Đặt Nhanh (1 Lệnh)

### Cách 1: Chạy setup.bat (Khuyến nghị)
```bash
setup.bat
```

Script sẽ tự động:
1. ✅ Yêu cầu quyền Administrator
2. ✅ Kiểm tra và cài Python (nếu chưa có)
3. ✅ Cài đặt các thư viện Python
4. ✅ Tải và cài VB-Cable
5. ✅ Cấu hình mọi thứ

### Cách 2: Chạy từ PowerShell
```powershell
powershell -Command "Start-Process setup.bat -Verb RunAs"
```

## Xử Lý Lỗi Tự Động

Script đã được cải tiến để thử 4 phương pháp cài đặt:

### Method 1: PowerShell với Admin Elevation
- Phương pháp đáng tin cậy nhất
- Tự động yêu cầu quyền admin
- Chạy im lặng (silent install)

### Method 2: ShellExecute API
- Sử dụng Windows API trực tiếp
- Tự động hiển thị UAC prompt
- Backup cho Method 1

### Method 3: Subprocess với Runas
- Sử dụng lệnh runas của Windows
- Thử với tài khoản Administrator

### Method 4: Installer với UI
- Hiển thị giao diện installer
- Cho phép người dùng thấy tiến trình
- Phương pháp cuối cùng nếu các cách khác thất bại

## Lỗi 4294967295 (0xFFFFFFFF)

Lỗi này thường do:
- ❌ Thiếu quyền admin → **Script tự động xử lý**
- ❌ Windows chặn driver → **Script thử nhiều cách**
- ❌ Antivirus chặn → **Tắt tạm thời antivirus**

### Giải pháp tự động:
Script mới sẽ:
1. Tự động yêu cầu quyền admin
2. Thử 4 phương pháp cài đặt khác nhau
3. Chờ đợi và verify cài đặt
4. Hướng dẫn nếu tất cả đều thất bại

## Các Tính Năng Tự Động

### ✅ Auto Admin Elevation
```batch
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
)
```

### ✅ Auto Download VB-Cable
```python
urllib.request.urlretrieve(
    "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip",
    "vbcable.zip"
)
```

### ✅ Auto Extract & Install
```python
# Extract
zipfile.ZipFile(zip_path).extractall(temp_dir)

# Install with multiple methods
PowerShell → ShellExecute → Runas → UI
```

### ✅ Auto Verify
```python
# Check if installed
for device in audio_devices:
    if 'CABLE' in device.name:
        return True
```

## Nếu Vẫn Gặp Lỗi

### 1. Tắt Antivirus tạm thời
```
Windows Security → Virus & threat protection → 
Manage settings → Real-time protection → OFF
```

### 2. Chạy lại setup.bat
```bash
setup.bat
```

### 3. Kiểm tra Windows Defender
```
Windows Security → App & browser control → 
Reputation-based protection settings → 
Check apps and files → OFF (tạm thời)
```

### 4. Cài thủ công (nếu cần)
```bash
# Giải nén vbcable_installer.zip
# Chuột phải VBCABLE_Setup_x64.exe
# Run as Administrator
# Click "Install Driver"
```

## Sau Khi Cài Đặt

### 1. Khởi động lại máy tính (BẮT BUỘC)
```
Restart → Required for driver to load
```

### 2. Chạy ứng dụng
```bash
run.bat
```

### 3. Cấu hình Audio
```
Click "⚙️ Audio Setup"
Select "CABLE Input (VB-Audio Virtual Cable)"
Click "▶️ Start Routing"
```

### 4. Cấu hình Discord
```
Settings → Voice & Video
Input Device → "CABLE Output (VB-Audio Virtual Cable)"
```

## Kiểm Tra Cài Đặt

### Kiểm tra VB-Cable đã cài chưa:
```bash
python -c "from src.virtual_audio import virtual_audio; print('Installed!' if virtual_audio.is_installed() else 'Not installed')"
```

### Liệt kê audio devices:
```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

## Troubleshooting

### Lỗi: "User may have cancelled UAC prompt"
**Giải pháp:** Nhấn "Yes" khi UAC prompt xuất hiện

### Lỗi: "Installation taking longer than expected"
**Giải pháp:** Đợi thêm 1-2 phút, đây là bình thường

### Lỗi: "All installation methods failed"
**Giải pháp:** 
1. Tắt antivirus
2. Chạy lại setup.bat
3. Hoặc cài thủ công theo hướng dẫn

## Liên Hệ

Nếu vẫn gặp vấn đề, vui lòng:
1. Chụp màn hình lỗi
2. Gửi file log (nếu có)
3. Mô tả chi tiết vấn đề

---

**Lưu ý:** Script đã được tối ưu để tự động 100%, nhưng Windows UAC vẫn cần người dùng nhấn "Yes" một lần.
