import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image using OpenCV (BGR format)
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//4.1.04.tiff'  # Change if needed
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Convert to RGB for matplotlib display
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

# Ripple parameters
amplitude = 5       # pixels
frequency = 0.05    # ripple density
phase = 0           # phase shift
center_x, center_y = width // 2, height // 2

# Generate coordinate grid
Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2)
r_safe = r + 1e-6  # avoid divide-by-zero
offset = amplitude * np.sin(frequency * r + phase)

# Compute new coordinates
map_x = X + (dx / r_safe) * offset
map_y = Y + (dy / r_safe) * offset

# Remap image using the ripple coordinates
rippled = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Show original and rippled images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(rippled)
axes[1].set_title('Ripple Effect (OpenCV)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
