import cv2
import matplotlib.pyplot as plt

# Step 1: Load grayscale image
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//4.1.02.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise FileNotFoundError(f"{image_path} not found!")

# Step 2: Apply pseudo-coloring (Jet colormap)
pseudo_colored = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
pseudo_colored = cv2.cvtColor(pseudo_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Step 3: Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(pseudo_colored)
axes[1].set_title('Pseudo Colored (Jet)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
