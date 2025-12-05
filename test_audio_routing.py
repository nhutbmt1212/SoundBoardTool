"""Test audio routing trực tiếp đến VB-Cable"""
import pyaudio
import numpy as np
import time
import wave
import os

def list_devices():
    """Liệt kê tất cả audio devices"""
    p = pyaudio.PyAudio()
    print("=" * 60)
    print("DANH SÁCH AUDIO DEVICES")
    print("=" * 60)
    
    cable_input_idx = None
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name']
        out_ch = info['maxOutputChannels']
        in_ch = info['maxInputChannels']
        
        marker = ""
        name_lower = name.lower()
        
        # Detect VB-Cable devices
        if 'vb-audio' in name_lower or 'cable' in name_lower:
            if out_ch > 0:  # Có output = có thể gửi audio vào
                if 'input' in name_lower or 'speakers' in name_lower:
                    marker = " ← DÙNG CHO SOUNDBOARD (output)"
                    if cable_input_idx is None:
                        cable_input_idx = i
                else:
                    marker = " ← VB-Audio Output Device"
            elif in_ch > 0:  # Chỉ có input = mic ảo
                marker = " ← DÙNG CHO DISCORD (input/mic)"
        
        if out_ch > 0 or in_ch > 0:
            print(f"[{i}] {name}")
            print(f"    Output channels: {out_ch}, Input channels: {in_ch}{marker}")
    
    p.terminate()
    return cable_input_idx

def test_tone(device_index, duration=3):
    """Phát tone test đến device"""
    print(f"\n🔊 Phát tone 440Hz đến device index {device_index} trong {duration} giây...")
    
    p = pyaudio.PyAudio()
    
    # Lấy thông tin device
    info = p.get_device_info_by_index(device_index)
    print(f"   Device: {info['name']}")
    print(f"   Default sample rate: {int(info['defaultSampleRate'])}")
    
    # Dùng sample rate của device
    sample_rate = int(info['defaultSampleRate'])
    print(f"   Using sample rate: {sample_rate}")
    
    # Tạo tone 440Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz, 50% volume
    
    # Convert to int16
    audio_data = (tone * 32767).astype(np.int16)
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_index
        )
        
        # Phát audio
        stream.write(audio_data.tobytes())
        
        stream.stop_stream()
        stream.close()
        print("✅ Đã phát xong!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    
    p.terminate()

def test_wav_file(device_index, wav_path):
    """Phát file WAV đến device"""
    if not os.path.exists(wav_path):
        print(f"❌ File không tồn tại: {wav_path}")
        return
    
    print(f"\n🔊 Phát file {wav_path} đến device index {device_index}...")
    
    p = pyaudio.PyAudio()
    
    try:
        wf = wave.open(wav_path, 'rb')
        
        print(f"   Channels: {wf.getnchannels()}")
        print(f"   Sample rate: {wf.getframerate()}")
        print(f"   Sample width: {wf.getsampwidth()}")
        
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=device_index
        )
        
        chunk = 1024
        data = wf.readframes(chunk)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        
        stream.stop_stream()
        stream.close()
        wf.close()
        print("✅ Đã phát xong!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    
    p.terminate()

if __name__ == "__main__":
    cable_idx = list_devices()
    
    print("\n" + "=" * 60)
    
    if cable_idx is not None:
        print(f"\n✅ Tìm thấy VB-Audio device tại index: {cable_idx}")
        input("\nNhấn Enter để phát tone test...")
        test_tone(cable_idx, duration=5)
    else:
        print("\n⚠️ Không tự động tìm thấy VB-Audio device")
    
    # Cho phép test thủ công bất kỳ device nào
    print("\n📋 Bạn có thể test thủ công bất kỳ device nào")
    device_idx = input("Nhập device index để test (hoặc Enter để thoát): ")
    if device_idx:
        test_tone(int(device_idx), duration=5)
        print("\n⚠️  Kiểm tra Discord Mic Test xem có nhận được không!")
