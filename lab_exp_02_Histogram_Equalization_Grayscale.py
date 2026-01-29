import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_ubyte

# Load the grayscale image (already in grayscale)
img_gray = io.imread('D://DIP_Handout//DIP_LAB2//DIP_LAB//Dataset//7.2.01.tiff')

# Ensure image is in uint8 format [0, 255]
img_gray = img_as_ubyte(img_gray)

# Perform histogram equalization
img_eq_gray = exposure.equalize_hist(img_gray)
img_eq_gray = img_as_ubyte(img_eq_gray)  # Convert back to [0, 255] format

# Plot images and histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Display original grayscale image
axes[0, 0].imshow(img_gray, cmap='gray')
axes[0, 0].set_title("Original Grayscale")
axes[0, 0].axis('off')

# Display equalized grayscale image
axes[0, 1].imshow(img_eq_gray, cmap='gray')
axes[0, 1].set_title("Equalized Grayscale")
axes[0, 1].axis('off')

# Plot histogram of original image
axes[1, 0].hist(img_gray.ravel(), bins=256, range=(0, 255), histtype='step', color='black')
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 255)

# Plot histogram of equalized image
axes[1, 1].hist(img_eq_gray.ravel(), bins=256, range=(0, 255), histtype='step', color='black')
axes[1, 1].set_title("Equalized Histogram")
axes[1, 1].set_xlim(0, 255)

plt.tight_layout()
plt.show()