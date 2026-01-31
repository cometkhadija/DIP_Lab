import numpy as np
import cv2
import pywt
from skimage.data import astronaut
import matplotlib.pyplot as plt

# Load astronaut image from skimage
rgb = astronaut()

# Split channels
channels = cv2.split(rgb)

# Function to get DWT subband for one channel
def get_dwt_channel(channel):
    cA, (cH, cV, cD) = pywt.dwt2(channel, 'db2')
    rows, cols = cA.shape
    combined = np.zeros((rows*2, cols*2), dtype=np.float32)
    combined[0:rows, 0:cols] = cA
    combined[0:rows, cols:] = cH
    combined[rows:, 0:cols] = cV
    combined[rows:, cols:] = cD
    return cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply DWT on each channel
dwt_channels = [get_dwt_channel(ch) for ch in channels]

# Merge back to RGB
dwt_rgb = cv2.merge(dwt_channels)

# Resize original to match
resized = cv2.resize(rgb, (dwt_rgb.shape[1], dwt_rgb.shape[0]))

# Plot
plt.subplot(1, 2, 1)
plt.imshow(resized)
plt.title('Original (Resized)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dwt_rgb)
plt.title('DWT RGB Subbands')
plt.axis('off')

plt.tight_layout()
plt.show()
