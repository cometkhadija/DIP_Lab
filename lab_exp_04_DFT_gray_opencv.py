import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread(cv2.samples.findFile("E://4-1//DIP Lab//Tutorial_2//misc//5.1.13.tiff"), cv2.IMREAD_GRAYSCALE)  # Replace with 'camera' equivalent
if image is None:
    raise FileNotFoundError("Image not found.")

# Convert image to float32 for DFT
image_float = np.float32(image)

# Compute 2D DFT using OpenCV
dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency component to the center
dft_shifted = np.fft.fftshift(dft)

# Compute magnitude spectrum
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])

# Apply log transform for visualization
magnitude_log = np.log1p(magnitude)

# Normalize to [0, 1] for display
magnitude_norm = cv2.normalize(magnitude_log, None, 0.0, 1.0, cv2.NORM_MINMAX)

# Plot original and spectrum
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(magnitude_norm, cmap='gray')
axes[1].set_title('DFT Magnitude Spectrum')
axes[1].axis('off')

plt.tight_layout()
plt.show()
