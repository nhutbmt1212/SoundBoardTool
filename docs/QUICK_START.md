# ⚡ Quick Start Guide

## 🎯 Mục tiêu
Cho phép mọi người trong Discord/Game nghe được soundboard của bạn trong 3 phút!

## 📋 Checklist

### ✅ Bước 1: Cài đặt tự động (1 phút)
```bash
# Windows: Double-click hoặc chạy
setup.bat

# Hoặc
python setup.py

# Restart máy tính sau khi cài xong
```

### ✅ Bước 2: Chạy app (30 giây)
```bash
python main.py
```

### ✅ Bước 3: Thêm sounds (30 giây)
1. Click "➕ Add Sound"
2. Chọn file âm thanh (.wav, .mp3, .ogg)
3. Test bằng cách click vào button

### ✅ Bước 4: Setup Audio Routing (1 phút)

**4.1. Cấu hình Soundboard**
1. Click "⚙️ Audio Setup"
2. Chọn "CABLE Input (VB-Audio Virtual Cable)"
3. Click "▶️ Start Routing"

**4.2. Cấu hình Discord**
1. Discord Settings → Voice & Video
2. Input Device → Chọn "CABLE Output (VB-Audio Virtual Cable)"
3. Test: Phát sound và xem thanh xanh

### ✅ Done! 🎉

## 🎮 Sử dụng

```
1. Mở Discord/Game
2. Join voice channel
3. Phát sound từ soundboard
4. Mọi người sẽ nghe được!
```

## ⚠️ Lưu ý

**Bạn sẽ không nghe thấy mic thật của mình!**

Giải pháp:
- Dùng Voicemeeter để mix mic + soundboard
- Hoặc dùng 2 Discord accounts (1 cho soundboard, 1 cho mic)

## 🆘 Gặp vấn đề?

| Vấn đề | Giải pháp |
|--------|-----------|
| Auto-install không chạy | Chạy `setup.bat` as Administrator |
| Không cài được pygame | `pip install pygame-ce` |
| Không cài được pyaudio | `pip install pipwin` → `pipwin install pyaudio` |
| Không thấy CABLE trong list | Khởi động lại máy sau khi setup |
| Discord không nhận âm thanh | Kiểm tra Input Device = "CABLE Output" |
| Âm thanh bị lag | Giảm buffer size, đóng app khác |

## 📖 Đọc thêm

- Chi tiết: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Full docs: [README.md](README.md)

---

**Thời gian setup**: ~3 phút (với auto-install)  
**Độ khó**: ⭐☆☆☆☆ (Rất dễ)  
**Kết quả**: Troll bạn bè cực mạnh! 😎

---

💡 **Pro Tip**: Chạy `setup.bat` một lần là xong tất cả!
