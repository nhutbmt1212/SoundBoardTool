# Virtual Audio Device Guide

## What is Virtual Audio Device?

A Virtual Audio Device creates a "virtual cable" between your soundboard and Discord/Games.

```
Soundboard → Virtual Device Input → Virtual Device Output → Discord/Game
```

## How It Works

### Without Virtual Audio:
- Soundboard plays to your speakers
- Discord hears your microphone
- ❌ Discord doesn't hear soundboard

### With Virtual Audio:
- Soundboard plays to Virtual Device
- Discord listens to Virtual Device
- ✅ Discord hears soundboard!

## Installation

### Automatic (Recommended)
1. Run SoundboardPro.exe
2. First run will ask: "Install Virtual Audio Device?"
3. Click "Yes"
4. Wait 1-2 minutes
5. Restart computer
6. Done!

### Manual
1. Download VB-Cable: https://vb-audio.com/Cable/
2. Extract and run setup
3. Restart computer
4. Run SoundboardPro.exe

## Setup

### In Soundboard Pro:
1. Click "⚙️ Audio Setup"
2. Select "CABLE Input (VB-Audio Virtual Cable)"
3. Click "▶️ Start Routing"

### In Discord:
1. Settings → Voice & Video
2. Input Device → "CABLE Output (VB-Audio Virtual Cable)"
3. Test by playing a sound

### In Games:
Similar to Discord - set microphone to "CABLE Output"

## Device Names

### CABLE Input
- **Use in**: Soundboard Pro
- **Purpose**: Where soundboard sends audio
- **Think of it as**: The "input" to the virtual cable

### CABLE Output
- **Use in**: Discord/Games
- **Purpose**: Where Discord/Games listen
- **Think of it as**: The "output" from the virtual cable

## Troubleshooting

### "CABLE Input" not in device list
**Solution:**
- Restart computer after installation
- Restart SoundboardPro.exe
- Check VB-Cable is installed in Control Panel

### Discord doesn't hear sounds
**Solution:**
1. Check Discord Input Device = "CABLE Output"
2. Check Input Volume is not muted
3. Ensure "▶️ Start Routing" is active
4. Try leaving and rejoining voice channel

### Can't hear my own microphone
**Problem:** When using CABLE Output as input, Discord only hears soundboard

**Solution Option 1: Use Voicemeeter**
- Download Voicemeeter: https://vb-audio.com/Voicemeeter/
- Mix your real microphone + soundboard
- More complex but more flexible

**Solution Option 2: Two Discord Accounts**
- One account for soundboard (CABLE Output)
- One account for your voice (real microphone)
- Simple but requires two accounts

**Solution Option 3: Push-to-Talk**
- Use soundboard when not talking
- Use real mic when talking
- Switch between them

### Sounds play to speakers instead of Discord
**Solution:**
- Check "▶️ Start Routing" is active in soundboard
- Check correct device selected in Audio Setup
- Restart soundboard

## Advanced: Voicemeeter Setup

For mixing real microphone + soundboard:

1. Install Voicemeeter Banana
2. Set Hardware Input 1 = Your real microphone
3. Set Hardware Input 2 = CABLE Output
4. Set Hardware Output = Your speakers
5. In Discord: Input Device = VoiceMeeter Output
6. In Soundboard: Output Device = CABLE Input

Now Discord hears both your mic and soundboard!

## Uninstallation

### Remove VB-Cable:
1. Go to Control Panel → Programs
2. Find "VB-Audio Virtual Cable"
3. Uninstall
4. Restart computer

Or:

1. Go to: `C:\Program Files\VB\CABLE\`
2. Run: `VBCABLE_Setup_x64.exe`
3. Click "Remove Driver"
4. Restart computer

## FAQ

**Q: Do I need VB-Cable?**
A: Only if you want Discord/Games to hear your soundboard. Otherwise, soundboard works fine without it.

**Q: Is VB-Cable safe?**
A: Yes, it's a legitimate audio driver from VB-Audio Software.

**Q: Can I use other virtual audio devices?**
A: Yes! Voicemeeter, Virtual Audio Cable, etc. all work.

**Q: Why does it need a restart?**
A: Windows needs to load the new audio driver.

**Q: Can I use soundboard without restarting?**
A: Yes, but audio routing won't work until restart.

**Q: Will this affect my other audio?**
A: No, it just adds a new virtual device. Your real devices work normally.

---

**Need more help?** Check the [User Guide](USER_GUIDE.md)
