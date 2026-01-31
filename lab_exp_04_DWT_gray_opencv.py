import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def draw_border(img, top, left, h, w, thickness=3, color=255):
    """Draws a rectangular border around a subband in the image."""
    img[top:top+thickness, left:left+w] = color
    img[top+h-thickness:top+h, left:left+w] = color
    img[top:top+h, left:left+thickness] = color
    img[top:top+h, left+w-thickness:left+w] = color

# Load the classic "cameraman" image via PyWavelets
camera_img = pywt.data.camera().astype(np.float32)

# Apply 2D DWT (Daubechies-2)
cA, (cH, cV, cD) = pywt.dwt2(camera_img, 'db2')

# Combine subbands into a 2x2 layout
rows, cols = cA.shape
combined = np.zeros((rows * 2, cols * 2), dtype=cA.dtype)

combined[0:rows,    0:cols]    = cA
combined[0:rows,    cols:2*cols]= cH
combined[rows:2*rows, 0:cols]  = cV
combined[rows:2*rows, cols:2*cols]= cD

# Draw borders around the four subbands
white_val = 255.0  # float image scale
for (top, left) in [(0, 0), (0, cols), (rows, 0), (rows, cols)]:
    draw_border(combined, top, left, rows, cols, thickness=3, color=white_val)

# Resize the original camera image to fit the combined layout
resized = cv2.resize(camera_img, (combined.shape[1], combined.shape[0]), interpolation=cv2.INTER_NEAREST)

# Display side by side using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(resized, cmap='gray')
axs[0].set_title('Original Image (Resized)')
axs[0].axis('off')

axs[1].imshow(combined, cmap='gray')
axs[1].set_title('Wavelet Subbands with Borders')
axs[1].axis('off')

plt.tight_layout()
plt.show()
