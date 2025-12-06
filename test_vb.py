import sounddevice as sd

print('=== VB-Cable Devices ===')
for i, d in enumerate(sd.query_devices()):
    name = d['name']
    if 'vb' in name.lower() or 'cable' in name.lower():
        print(f'[{i}] {name}')
        print(f'    Input: {d["max_input_channels"]}')
        print(f'    Output: {d["max_output_channels"]}')
        print()
