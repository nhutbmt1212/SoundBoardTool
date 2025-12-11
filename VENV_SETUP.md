# SoundBoardTool - Virtual Environment Setup

## 📦 Giống như Node.js node_modules

Project này giờ sử dụng **Python Virtual Environment (venv)** - tương tự như `node_modules` trong Node.js!

### ✅ Lợi ích:
- ✅ Thư viện được cài **riêng cho project** này
- ✅ Không ảnh hưởng đến Python global
- ✅ Dễ dàng quản lý dependencies
- ✅ Có thể xóa và tạo lại bất cứ lúc nào

## 🚀 Cách sử dụng

### Lần đầu tiên - Setup:
```bash
setup_env.bat
```
Script này sẽ:
1. Tạo virtual environment trong folder `venv/`
2. Cài đặt tất cả dependencies
3. Sẵn sàng để chạy!

### Chạy app:
```bash
run.bat
```
Script này sẽ:
1. Tự động activate virtual environment
2. Chạy app
3. Tự động deactivate khi thoát

### Thủ công (nếu cần):

**Activate venv:**
```bash
venv\Scripts\activate
```

**Cài thêm package:**
```bash
pip install package-name
```

**Deactivate:**
```bash
deactivate
```

## 📁 Cấu trúc

```
SoundBoardTool/
├── venv/                    # Virtual environment (giống node_modules)
│   ├── Scripts/            # Executables
│   ├── Lib/                # Python libraries
│   └── ...
├── src/                    # Source code
├── requirements.txt        # Main dependencies
├── requirements_backup.txt # Backup dependencies
├── setup_env.bat          # Setup script
└── run.bat                # Run script
```

## 🔧 Dependencies

### Main (`requirements.txt`):
- Các thư viện chính của app

### Backup (`requirements_backup.txt`):
- `google-auth` - Google authentication
- `google-auth-oauthlib` - OAuth flow
- `google-api-python-client` - Google Drive API

## 🗑️ Xóa và tạo lại

Nếu gặp vấn đề, bạn có thể xóa folder `venv/` và chạy lại `setup_env.bat`

```bash
# Xóa venv
rmdir /s /q venv

# Tạo lại
setup_env.bat
```

## 📝 Lưu ý

- ✅ Folder `venv/` đã được thêm vào `.gitignore`
- ✅ Không commit `venv/` lên Git
- ✅ Chỉ commit `requirements.txt` và `requirements_backup.txt`
- ✅ Người khác clone về chỉ cần chạy `setup_env.bat`

## 🎯 Workflow

1. **Clone project** → Chạy `setup_env.bat`
2. **Develop** → Chạy `run.bat` để test
3. **Add dependency** → Thêm vào `requirements.txt` hoặc `requirements_backup.txt`
4. **Commit** → Chỉ commit file requirements, không commit `venv/`

---

**Giờ project của bạn hoạt động giống Node.js với npm!** 🎉
