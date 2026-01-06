import tensorflow as tf
import cv2
import numpy as np

print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

# Test OpenCV contrib module
if hasattr(cv2, 'face'):
    print("cv2.face module is available ✅")
else:
    print("cv2.face module not found ❌")
