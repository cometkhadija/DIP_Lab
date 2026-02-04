import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//4.2.03.tiff'
rgb_image = cv2.imread(image_path)
if rgb_image is None:
    raise FileNotFoundError(f"{image_path} not found!")

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
rgb_image = rgb_image.astype(np.float32) / 255.0         # Normalize to [0,1]

# Step 2: Split into channels
R, G, B = cv2.split(rgb_image)

# Step 3: Apply Sobel edge detection to each channel
def sobel_edge(channel):
    grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x**2 + grad_y**2)

R_edge = sobel_edge(R)
G_edge = sobel_edge(G)
B_edge = sobel_edge(B)

# Step 4: Recombine edge-detected channels into RGB
edge_rgb = cv2.merge([R_edge, G_edge, B_edge])

# Step 5: Create grayscale edge map
edge_gray = (R_edge + G_edge + B_edge) / 3

# Step 6: Display results
fig, axes = plt.subplots(1, 6, figsize=(24, 5))

axes[0].imshow(rgb_image)
axes[0].set_title('Original RGB')
axes[0].axis('off')

axes[1].imshow(R_edge, cmap='gray')
axes[1].set_title('Red Channel Edges')
axes[1].axis('off')

axes[2].imshow(G_edge, cmap='gray')
axes[2].set_title('Green Channel Edges')
axes[2].axis('off')

axes[3].imshow(B_edge, cmap='gray')
axes[3].set_title('Blue Channel Edges')
axes[3].axis('off')

axes[4].imshow(edge_rgb)
axes[4].set_title('Combined Edge RGB')
axes[4].axis('off')

axes[5].imshow(edge_gray, cmap='gray')
axes[5].set_title('Grayscale Edge Map')
axes[5].axis('off')

plt.tight_layout()
plt.show()
