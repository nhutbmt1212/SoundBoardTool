"""Test script để kiểm tra VB-Cable"""
import pyaudio

print("=" * 50)
print("KIỂM TRA VB-CABLE")
print("=" * 50)

# 1. Kiểm tra qua PyAudio
print("\n📢 Danh sách Audio Devices:")
p = pyaudio.PyAudio()

found_cable = False
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    name = info['name']
    
    # Highlight nếu là VB-Cable
    if 'cable' in name.lower() or 'vb-audio' in name.lower():
        print(f"  ✅ [{i}] {name} (VB-CABLE FOUND!)")
        found_cable = True
    else:
        print(f"  [{i}] {name}")

p.terminate()

print("\n" + "=" * 50)
if found_cable:
    print("✅ VB-Cable ĐÃ CÀI!")
else:
    print("❌ VB-Cable CHƯA CÀI hoặc chưa được nhận diện")
    print("\nCó thể do:")
    print("  1. Chưa restart sau khi cài")
    print("  2. Driver chưa được load")
    print("  3. Cài bị lỗi")

# 2. Kiểm tra qua Registry
print("\n📋 Kiểm tra Registry:")
try:
    import winreg
    paths = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    ]
    
    found_reg = False
    for path in paths:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    subkey = winreg.OpenKey(key, subkey_name)
                    try:
                        name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        if "vb" in name.lower() or "cable" in name.lower() or "virtual" in name.lower():
                            print(f"  ✅ Found: {name}")
                            found_reg = True
                    except:
                        pass
                    winreg.CloseKey(subkey)
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except:
            continue
    
    if not found_reg:
        print("  ❌ Không tìm thấy trong Registry")
except Exception as e:
    print(f"  Lỗi: {e}")

print("\n" + "=" * 50)
