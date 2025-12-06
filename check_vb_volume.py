"""Check VB-Cable volume and mute status"""
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    
    print("=== VB-Cable Audio Status ===\n")
    
    # Get all devices
    devices = AudioUtilities.GetAllDevices()
    
    for device in devices:
        if 'cable' in device.FriendlyName.lower():
            print(f"Device: {device.FriendlyName}")
            print(f"  State: {device.state}")
            
            try:
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = interface.QueryInterface(IAudioEndpointVolume)
                
                current_volume = volume.GetMasterVolumeLevelScalar()
                is_muted = volume.GetMute()
                
                print(f"  Volume: {current_volume * 100:.0f}%")
                print(f"  Muted: {is_muted}")
            except Exception as e:
                print(f"  Cannot get volume: {e}")
            print()
            
except ImportError:
    print("pycaw not installed - cannot check volume")
    print("Install: pip install pycaw")
