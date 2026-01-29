import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, img_as_ubyte

# Load the RGB image
img_rgb = io.imread('D://DIP_Handout//DIP_LAB2//DIP_LAB//Dataset//4.1.02.tiff')

# Convert RGB to HSV
img_hsv = color.rgb2hsv(img_rgb)

# Extract the Value (V) channel and convert to uint8
v_channel = img_as_ubyte(img_hsv[:, :, 2])

# Perform histogram equalization on the V channel
v_eq = exposure.equalize_hist(v_channel)
v_eq = img_as_ubyte(v_eq)

# Replace the V channel with the equalized version (normalized to [0,1] for HSV)
img_hsv[:, :, 2] = v_eq / 255.0

# Convert back to RGB
img_eq_rgb = color.hsv2rgb(img_hsv)
img_eq_rgb = img_as_ubyte(img_eq_rgb)  # Ensure final image is in [0,255]

# Plot original and equalized images
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original RGB image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')

# Equalized RGB image
axes[0, 1].imshow(img_eq_rgb)
axes[0, 1].set_title("RGB via HSV Equalization")
axes[0, 1].axis('off')

# Histogram of original V channel
axes[1, 0].hist(v_channel.ravel(), bins=256, range=(0, 255), histtype='step', color='black')
axes[1, 0].set_title("Original V Channel Histogram")
axes[1, 0].set_xlim(0, 255)

# Histogram of equalized V channel
axes[1, 1].hist(v_eq.ravel(), bins=256, range=(0, 255), histtype='step', color='black')
axes[1, 1].set_title("Equalized V Channel Histogram")
axes[1, 1].set_xlim(0, 255)

plt.tight_layout()
plt.show()