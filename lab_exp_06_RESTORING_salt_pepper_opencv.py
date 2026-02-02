import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread(cv2.samples.findFile('E://4-1//DIP Lab//Tutorial_2//misc//4.1.04.tiff'), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

# Add salt & pepper noise
noisy = img.copy()
prob = 0.05
num_salt = int(prob * img.size / 2)
coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
noisy[coords] = 1
coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
noisy[coords] = 0

# Apply median filters
med3 = cv2.medianBlur((noisy*255).astype(np.uint8), 3)
med5 = cv2.medianBlur((noisy*255).astype(np.uint8), 5)

# Display results
titles = ['Original', 'Salt & Pepper Noise', 'Median Filter 3x3', 'Median Filter 5x5']
images = [img, noisy, med3/255.0, med5/255.0]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
