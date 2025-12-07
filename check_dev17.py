import sounddevice as sd
d = sd.query_devices(17)
print(f'Device 17: {d["name"]}')
print(f'Max output channels: {d["max_output_channels"]}')
print(f'Default sample rate: {d["default_samplerate"]}')
