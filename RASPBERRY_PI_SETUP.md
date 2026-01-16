# Raspberry Pi 4B Setup Guide (Python 3.13 + Debian)

## System Requirements
- Raspberry Pi 4B (4GB+ RAM recommended)
- Debian-based OS (Raspberry Pi OS Bookworm or later)
- Python 3.13
- CSI Camera Module or USB Webcam
- Microphone and Speaker

## Installation Steps

### 1. System Update
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install System Dependencies
```bash
# Camera support
sudo apt install -y libcamera-dev libcamera-apps python3-picamera2

# Audio support
sudo apt install -y portaudio19-dev python3-pyaudio alsa-utils

# OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv libatlas-base-dev

# Build tools
sudo apt install -y build-essential cmake pkg-config
```

### 3. Python Environment Setup
```bash
cd Pi_Buddy/pi_services

# Create virtual environment (optional but recommended)
python3.13 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Model Conversion (TFLite to ONNX)
If you have a TFLite model, convert it to ONNX:

```bash
# Install conversion tools
pip install tf2onnx tensorflow

# Convert model
python3 -m tf2onnx.convert --tflite models/model.tflite --output models/model.onnx
```

### 5. Camera Configuration
Enable camera interface:
```bash
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
```

### 6. Audio Configuration
Test microphone:
```bash
arecord -l  # List recording devices
aplay -l    # List playback devices
```

Configure default audio device in `/etc/asound.conf` if needed.

### 7. Environment Variables
Create `.env` file in `pi_services/` directory:
```bash
# Database (Neon PostgreSQL)
DB_HOST=your-neon-host.neon.tech
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
DB_PORT=5432

# LLM Service URL
LLM_SERVICE_URL=http://192.168.31.150:8000

# Camera settings (optional)
BUDDY_CAMERA_INDEX=0
BUDDY_LOG_LEVEL=WARNING
```

### 8. Performance Optimization for Pi 4B

#### CPU Governor
```bash
# Set performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### Memory Split
```bash
sudo raspi-config
# Advanced Options -> Memory Split -> Set to 256MB for GPU
```

#### Disable unnecessary services
```bash
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### 9. Run the Application
```bash
cd Pi_Buddy/pi_services
python3.13 buddy_pi.py
```

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
libcamera-hello --list-cameras

# Test camera
libcamera-still -o test.jpg
```

### Audio Issues
```bash
# Test microphone
arecord -d 5 test.wav
aplay test.wav

# Adjust volume
alsamixer
```

### ONNX Runtime Issues
If ONNX runtime fails, try:
```bash
pip install onnxruntime --no-cache-dir
```

### Memory Issues
Increase swap space:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Key Optimizations for Pi 4B

1. **Multi-threading**: ONNX uses 4 threads (Pi 4B has 4 cores)
2. **Camera buffering**: Minimal buffer size for low latency
3. **Frame processing**: Optimized intervals to prevent CPU overload
4. **Model inference**: ONNX with graph optimization enabled
5. **Memory management**: Efficient numpy array handling

## Notes

- CSI camera is preferred over USB for better performance
- Ensure adequate cooling (heatsink/fan) for sustained operation
- Use quality power supply (5V 3A minimum)
- Model file must be in ONNX format (not TFLite)
- Python 3.13 is fully supported with all dependencies
