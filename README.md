# Mouse-Control-Using-Eye-Gaze-and-Eye-Blinks-

Overview
This project implements a real-time, gesture-based mouse control system using facial expressions and hand gestures. Designed for accessibility and hands-free interaction, it replaces traditional input devices with intuitive gestures detected via a standard webcam. The system combines MediaPipe's Face Mesh and Hand modules for precise landmark detection, enabling comprehensive computer control without physical devices.

Key Features
Cursor Control: Track nose movement for smooth pointer navigation

Click Actions:

* Left-click → Left eye blink
* Right-click → Right eye blink
* Double-click → Simultaneous blink

Navigation:

* Vertical scrolling → Head tilts
* Tab switching → Smile detection

Hand Gestures:

✊ Fist → On-screen keyboard

👍 Thumb up → Paste (Ctrl+V)

✌️ Peace → Copy (Ctrl+C)

🤘 Rock on → Undo (Ctrl+Z)

✋ Open palm → Screenshot

Limitations & Future Work
*Current limitations:*
1. Smile detection false positives during speech
2. Eye fatigue in prolonged use

*Planned enhancements:*

1. Customizable gesture mappings

2. Machine learning-based adaptive thresholds

3. 3D gesture recognition using depth sensors
