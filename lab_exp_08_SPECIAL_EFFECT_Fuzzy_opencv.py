import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (use 'messi5.jpg' if available, otherwise replace with your own file)
img = cv2.imread("E://4-1//DIP Lab//Tutorial_2//misc//house.tiff")
if img is None:
    raise FileNotFoundError("Image not found! Please provide a valid image path.")

# Convert BGR (OpenCV default) to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get dimensions
h, w, c = img_rgb.shape

# Window size for neighborhood
win = 5
r = win // 2

# Pad the image using reflection
padded = cv2.copyMakeBorder(img_rgb, r, r, r, r, cv2.BORDER_REFLECT)

# Output image initialized
fuzzy = np.zeros_like(img_rgb)

# Apply random neighbor replacement
for y in range(h):
    for x in range(w):
        offset_y = np.random.randint(-r, r + 1)
        offset_x = np.random.randint(-r, r + 1)
        fuzzy[y, x] = padded[y + r + offset_y, x + r + offset_x]

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(fuzzy)
plt.title("Fuzzy Image (Random Neighborhood)")
plt.axis("off")

plt.tight_layout()
plt.show()
