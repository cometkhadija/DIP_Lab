import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image
image = cv2.imread(cv2.samples.findFile("E://4-1//DIP Lab//Tutorial_2//misc//4.2.03.tiff"))  # Use your own image path if needed
if image is None:
    raise FileNotFoundError("Image not found.")

# Convert BGR to RGB for display consistency
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize to [0, 1] float for DFT
image_norm = image_rgb.astype(np.float32) / 255.0

# Process each channel
fft_channels = []
for i in range(3):  # R, G, B channels
    channel = image_norm[:, :, i]

    # DFT
    dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Magnitude and log scale
    magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    log_magnitude = np.log1p(magnitude)

    # Normalize for display
    log_magnitude_norm = cv2.normalize(log_magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
    fft_channels.append(log_magnitude_norm)

# Merge channels back to RGB spectrum image
fft_rgb = cv2.merge(fft_channels)

# Plot original and spectrum
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(fft_rgb)
axes[1].set_title('DFT Magnitude Spectrum (RGB)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
