#!/usr/bin/env python3
# csi_camera_helper.py - SYSTEM PYTHON ONLY
from picamera2 import Picamera2
from time import sleep
import os

OUTPUT = "/tmp/csi_frame.jpg"

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": "BGR888", "size": (320, 240)}))
cam.start()
sleep(1)

print("CSI camera helper started")

try:
    while True:
        cam.capture_file(OUTPUT)
        sleep(0.05)
except KeyboardInterrupt:
    cam.stop()
    print("CSI camera helper stopped")
