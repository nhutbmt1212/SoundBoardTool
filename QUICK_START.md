# ⚡ Quick Start Guide

## 🎯 Mục tiêu
Cho phép mọi người trong Discord/Game nghe được soundboard của bạn trong 5 phút!

## 📋 Checklist

### ✅ Bước 1: Cài đặt cơ bản (2 phút)
```bash
# Clone/tải project về
# Cài dependencies
pip install -r requirements.txt

# Chạy app
python main.py
```

### ✅ Bước 2: Thêm sounds (1 phút)
1. Click "➕ Add Sound"
2. Chọn file âm thanh (.wav, .mp3, .ogg)
3. Test bằng cách click vào button

### ✅ Bước 3: Setup Virtual Audio (2 phút)

**3.1. Cài VB-Cable**
- Tải: https://vb-audio.com/Cable/
- Cài đặt → Khởi động lại máy

**3.2. Cấu hình Soundboard**
1. Click "⚙️ Audio Setup"
2. Chọn "CABLE Input (VB-Audio Virtual Cable)"
3. Click "▶️ Start Routing"

**3.3. Cấu hình Discord**
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
| Không cài được pygame | `pip install pygame-ce` |
| Không cài được pyaudio | `pip install pipwin` → `pipwin install pyaudio` |
| Không thấy CABLE trong list | Khởi động lại máy sau khi cài VB-Cable |
| Discord không nhận âm thanh | Kiểm tra Input Device = "CABLE Output" |
| Âm thanh bị lag | Giảm buffer size, đóng app khác |

## 📖 Đọc thêm

- Chi tiết: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Full docs: [README.md](README.md)

---

**Thời gian setup**: ~5 phút  
**Độ khó**: ⭐⭐☆☆☆ (Dễ)  
**Kết quả**: Troll bạn bè cực mạnh! 😎
