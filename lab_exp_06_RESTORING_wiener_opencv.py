import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load grayscale image
img = cv2.imread("E://4-1//DIP Lab//Tutorial_2//misc//5.1.13.tiff", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

# Add salt & pepper noise
noisy = img.copy()
prob = 0.05
num_salt = int(prob * img.size / 2)
coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
noisy[coords] = 1
coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
noisy[coords] = 0

# Apply Wiener filter
restored = wiener(noisy, (5, 5))

# Display
titles = ['Original', 'Noisy', 'Wiener Filtered']
images = [img, noisy, restored]
for i in range(3):
    plt.subplot(1, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i]), plt.axis('off')
plt.tight_layout(), plt.show()



