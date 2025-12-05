# 🎙️ Hướng dẫn Setup Audio Routing

## Mục đích
Cho phép mọi người trong Discord/Game nghe được âm thanh từ Soundboard của bạn.

## Cách hoạt động
Soundboard sẽ phát âm thanh qua một **Virtual Audio Device** (thiết bị âm thanh ảo), và bạn sẽ chọn thiết bị này làm microphone trong Discord/Game.

## Bước 1: Cài đặt Virtual Audio Cable

### Option 1: VB-Audio Virtual Cable (Miễn phí, Khuyến nghị)
1. Tải về: https://vb-audio.com/Cable/
2. Giải nén và chạy file `VBCABLE_Setup_x64.exe` (hoặc x86)
3. Click "Install Driver"
4. Khởi động lại máy tính

### Option 2: Voicemeeter (Miễn phí, Nhiều tính năng hơn)
1. Tải về: https://vb-audio.com/Voicemeeter/
2. Cài đặt Voicemeeter Banana hoặc Potato
3. Khởi động lại máy tính

## Bước 2: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

Nếu gặp lỗi với PyAudio trên Windows:
```bash
pip install pipwin
pipwin install pyaudio
```

Hoặc tải wheel file từ: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Bước 3: Cấu hình Soundboard

1. Chạy soundboard:
   ```bash
   python main.py
   ```

2. Click nút "⚙️ Audio Setup"

3. Chọn virtual device (VD: "CABLE Input" hoặc "VoiceMeeter Input")

4. Click "▶️ Start Routing"

## Bước 4: Cấu hình Discord

1. Mở Discord Settings → Voice & Video
2. Trong "Input Device", chọn:
   - **CABLE Output** (nếu dùng VB-Cable)
   - **VoiceMeeter Output** (nếu dùng Voicemeeter)
3. Test microphone - bạn sẽ thấy thanh xanh khi phát sound

## Bước 5: Cấu hình Game

Tương tự Discord, vào settings của game và chọn virtual device làm microphone.

## Lưu ý quan trọng

### ⚠️ Bạn sẽ không nghe thấy microphone thật của mình
Khi dùng virtual cable, Discord/Game chỉ nghe được soundboard. Để nghe cả mic thật:

**Giải pháp 1: Dùng Voicemeeter (Khuyến nghị)**
- Voicemeeter cho phép mix nhiều audio sources
- Bạn có thể mix mic thật + soundboard

**Giải pháp 2: Dùng Windows Audio Mixer**
1. Right-click icon loa → Sounds → Recording
2. Right-click "CABLE Output" → Properties → Listen
3. Check "Listen to this device"
4. Select playback device

## Troubleshooting

### Không thấy Virtual Device trong list
- Đảm bảo đã cài VB-Cable hoặc Voicemeeter
- Khởi động lại máy tính
- Khởi động lại soundboard app

### Discord không nhận âm thanh
- Kiểm tra Input Device trong Discord settings
- Kiểm tra Input Volume không bị mute
- Test bằng cách phát sound và xem thanh xanh

### Âm thanh bị lag/delay
- Giảm buffer size trong audio settings
- Đóng các app khác đang dùng audio
- Cập nhật driver âm thanh

### PyAudio không cài được
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Hoặc dùng conda
conda install pyaudio
```

## Sơ đồ luồng âm thanh

```
Soundboard App
    ↓
Virtual Audio Device (CABLE Input / VoiceMeeter Input)
    ↓
Discord/Game (chọn CABLE Output / VoiceMeeter Output làm mic)
    ↓
Mọi người nghe được! 🎉
```

## Tips

1. **Điều chỉnh volume**: Dùng slider trong soundboard để tránh quá to
2. **Hotkeys**: Có thể thêm hotkeys để phát sound nhanh hơn
3. **Mix với mic**: Dùng Voicemeeter để mix soundboard + mic thật
4. **Test trước**: Test với bạn bè trước khi dùng trong game quan trọng

## Liên kết hữu ích

- VB-Audio Virtual Cable: https://vb-audio.com/Cable/
- Voicemeeter: https://vb-audio.com/Voicemeeter/
- PyAudio Wheels: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
