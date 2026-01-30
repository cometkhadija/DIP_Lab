import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import data, img_as_float, exposure

# Load grayscale Cameraman image
img = img_as_float(data.camera())

# --- 1. Averaging Filter (3x3 Box Blur)
avg_kernel = np.ones((3, 3), dtype=float) / 9.0
avg_filtered = convolve(img, avg_kernel)

# --- 2. Gaussian Filter (3x3 with σ ~ 1)
gauss_kernel = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=float)
gauss_kernel /= gauss_kernel.sum()  # Normalize kernel
gauss_filtered = convolve(img, gauss_kernel)

# --- 3. High-Pass Filter (Laplacian-like 3x3)
highpass_kernel = np.array([[0, -1,  0],
                            [-1,  4, -1],
                            [0, -1,  0]], dtype=float)
highpass_filtered = convolve(img, highpass_kernel)

# Rescale high-pass output for display
highpass_display = exposure.rescale_intensity(highpass_filtered, in_range='image', out_range=(0, 1))

# Plotting Results
titles = ['Original', 'Averaging (3x3)', 'Gaussian (3x3)', 'High-Pass (3x3)']
images = [img, avg_filtered, gauss_filtered, highpass_display]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()