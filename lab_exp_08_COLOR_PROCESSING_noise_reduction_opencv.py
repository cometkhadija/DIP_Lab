import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//house.tiff'
rgb_image = cv2.imread(image_path)
if rgb_image is None:
    raise FileNotFoundError(f"{image_path} not found!")

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
rgb_float = rgb_image.astype(np.float32) / 255.0         # Normalize to [0,1]

# Step 2: Add salt & pepper noise
def salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = np.ceil(amount * total_pixels)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 1
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    return noisy


noisy_image = salt_pepper_noise(rgb_float, amount=0.05)

# Step 3: Split channels and apply median filter
R_noisy, G_noisy, B_noisy = cv2.split(noisy_image)
R_denoised = cv2.medianBlur((R_noisy*255).astype(np.uint8), 5) / 255.0
G_denoised = cv2.medianBlur((G_noisy*255).astype(np.uint8), 5) / 255.0
B_denoised = cv2.medianBlur((B_noisy*255).astype(np.uint8), 5) / 255.0

# Step 4: Merge channels
denoised_image = cv2.merge([R_denoised, G_denoised, B_denoised])

# Step 5: Display results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(rgb_float)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(noisy_image)
axes[1].set_title('Noisy RGB Image')
axes[1].axis('off')

axes[2].imshow(denoised_image)
axes[2].set_title('Denoised RGB Image (Median Filter)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
