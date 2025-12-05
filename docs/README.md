# 🎵 Soundboard Pro

Một công cụ soundboard chuyên nghiệp với giao diện hiện đại, hỗ trợ routing âm thanh đến Discord/Games.

## ✨ Tính năng

### Cơ bản
- 🎨 Giao diện đẹp mắt với theme tối hiện đại
- 🎵 Phát các file âm thanh với một cú click
- ➕ Thêm file âm thanh mới dễ dàng
- 🔊 Điều chỉnh volume realtime
- ⏹️ Dừng tất cả âm thanh đang phát
- 🔄 Tự động refresh danh sách sounds

### Nâng cao
- 🎙️ **Audio Routing** - Route âm thanh đến Discord/Games
- 🎮 Cho phép mọi người trong voice chat nghe được soundboard
- 🔌 Hỗ trợ Virtual Audio Devices (VB-Cable, Voicemeeter)
- 📊 Hiển thị danh sách audio devices
- ⚙️ Cấu hình audio routing dễ dàng

## 📦 Cài đặt

### ⚡ Cài đặt tự động (Khuyến nghị)

**Windows - Chỉ cần 1 lệnh:**

```bash
# Double-click file này:
setup.bat

# Hoặc chạy:
python setup.py
```

Script sẽ tự động:
- ✅ Cài tất cả Python dependencies
- ✅ Tải và cài VB-Audio Virtual Cable
- ✅ Setup mọi thứ sẵn sàng

**Sau khi cài xong:**
1. Restart máy tính
2. Chạy: `python main.py`
3. Enjoy! 🎉

📖 **Chi tiết**: Xem [INSTALL.md](INSTALL.md)

---

### 🔧 Cài đặt thủ công (Nếu auto-install không hoạt động)

**Bước 1: Cài Python dependencies**
```bash
pip install -r requirements.txt
```

**Bước 2: Cài Virtual Audio Cable**
- Tải: https://vb-audio.com/Cable/
- Cài đặt và restart máy

**Lưu ý:** Nếu gặp lỗi PyAudio trên Windows:
```bash
pip install pipwin
pipwin install pyaudio
```

## 🚀 Sử dụng

### Chạy ứng dụng

**Windows - Cách dễ nhất:**
```bash
# Double-click:
run.bat
```

**Hoặc dùng Python:**
```bash
python main.py
```

### Sử dụng cơ bản

1. **Thêm sounds**: Click "➕ Add Sound" và chọn file âm thanh (.wav, .mp3, .ogg, .flac)
2. **Phát sound**: Click vào button của sound muốn phát
3. **Điều chỉnh volume**: Kéo slider "🔊 Volume"
4. **Dừng tất cả**: Click "⏹️ Stop All"

### 🎙️ Setup Audio Routing (Discord/Games)

Để mọi người trong Discord/Game nghe được soundboard:

1. **Cài Virtual Audio Cable** (xem Bước 3 ở trên)

2. **Mở Audio Setup**: Click "⚙️ Audio Setup" trong app

3. **Chọn Virtual Device**: 
   - Chọn "CABLE Input" (VB-Cable) hoặc "VoiceMeeter Input"
   - Click "▶️ Start Routing"

4. **Cấu hình Discord/Game**:
   - Mở Settings → Voice & Video
   - Chọn "CABLE Output" hoặc "VoiceMeeter Output" làm **Input Device**
   - Test bằng cách phát sound

5. **Done!** Mọi người giờ sẽ nghe được soundboard của bạn! 🎉

📖 **Xem hướng dẫn chi tiết**: [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 📁 Cấu trúc dự án

```
soundboard/
├── main.py              # Entry point
├── soundboard.py        # Core audio logic
├── ui.py               # Modern GUI interface
├── audio_router.py     # Audio routing system
├── config.py           # Configuration
├── sounds/             # Audio files folder
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── SETUP_GUIDE.md     # Detailed setup guide
```

## 🎮 Use Cases

- **Gaming**: Phát sound effects trong game với bạn bè
- **Streaming**: Thêm sound effects vào stream
- **Discord**: Troll bạn bè với meme sounds
- **Presentations**: Thêm sound effects vào thuyết trình
- **Podcasting**: Sound effects cho podcast

## 🛠️ Yêu cầu hệ thống

- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.7+
- **RAM**: 100MB+
- **Disk**: 50MB+ (không tính audio files)

## 📚 Dependencies

- `pygame-ce` - Audio playback
- `pyaudio` - Audio routing (optional)
- `numpy` - Audio processing (optional)
- `tkinter` - GUI (built-in với Python)

## 🐛 Troubleshooting

### Không cài được pygame
```bash
pip install pygame-ce
```

### Không cài được pyaudio
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install python3-pyaudio
```

### Không thấy Virtual Device
- Đảm bảo đã cài VB-Cable hoặc Voicemeeter
- Khởi động lại máy tính
- Khởi động lại app

### Discord không nhận âm thanh
- Kiểm tra Input Device trong Discord settings
- Đảm bảo đã chọn đúng "CABLE Output" hoặc "VoiceMeeter Output"
- Kiểm tra Input Volume không bị mute

## 💡 Tips

1. **Tổ chức sounds**: Đặt tên file rõ ràng để dễ tìm
2. **Volume control**: Điều chỉnh volume phù hợp để không quá to
3. **Hotkeys**: Có thể thêm hotkeys cho sounds hay dùng
4. **Mix với mic**: Dùng Voicemeeter để mix soundboard + mic thật

## 🔮 Tính năng sắp tới

- [ ] Hotkeys support
- [ ] Sound categories/folders
- [ ] Favorites system
- [ ] Search/filter sounds
- [ ] Custom button colors
- [ ] Sound preview
- [ ] Export/import sound packs
- [ ] Waveform visualization

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📞 Support

Nếu gặp vấn đề, xem [SETUP_GUIDE.md](SETUP_GUIDE.md) hoặc tạo issue trên GitHub.

---

Made with ❤️ | Happy Sound Boarding! 🎵
