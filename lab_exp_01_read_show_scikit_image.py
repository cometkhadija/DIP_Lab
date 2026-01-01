import matplotlib.pyplot as plt
from skimage import io, color

# Load the image using scikit-image
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
rgb_image = io.imread(image_path)  # Automatically loads as RGB if applicable

# Convert RGB to grayscale using scikit-image
gray_image = color.rgb2gray(rgb_image)  # Returns float image in range [0, 1]

# Plot both images
plt.figure(figsize=(10, 5))

# RGB image
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('RGB Image')
plt.axis('off')

# Grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()