"""
Sistem Python (3.13, picamera2) ile çalışır.
640x480 BGR kareleri stdout'a ham byte olarak yazar.
main.py tarafından subprocess ile başlatılır.
"""
import sys
import numpy as np
from picamera2 import Picamera2
import logging
logging.getLogger().setLevel(logging.ERROR)

FRAME_W, FRAME_H = 640, 480

picam2 = Picamera2()
# RGB888 libcamera'da bellekte BGR byte sırasıyla saklanır — OpenCV ile doğrudan uyumlu.
config = picam2.create_video_configuration(
    main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

FRAME_BYTES = FRAME_W * FRAME_H * 3

while True:
    frame = picam2.capture_array()
    sys.stdout.buffer.write(frame.tobytes())
    sys.stdout.buffer.flush()
