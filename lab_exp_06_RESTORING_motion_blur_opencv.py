import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread(cv2.samples.findFile('E://4-1//DIP Lab//Tutorial_2//misc//4.1.05.tiff'), cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

# Motion blur kernel
def motion_blur(size, angle):
    k = np.zeros((size, size))
    k[size//2, :] = 1
    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
    k = cv2.warpAffine(k, M, (size, size))
    return k / np.sum(k)

psf = motion_blur(15, 30)
blurred = cv2.filter2D(img, -1, psf)
noisy = blurred + 0.03 * np.random.randn(*blurred.shape)
noisy = np.clip(noisy, 0, 1)

# Inverse filter
def inverse_filter(img, psf, eps=1e-3):
    F, H = fft2(img), fft2(psf, s=img.shape)
    H[np.abs(H) < eps] = eps
    return np.real(ifft2(F / H))

# Wiener filter
def wiener_filter(img, psf, K=0.01):
    F, H = fft2(img), fft2(psf, s=img.shape)
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + K)) * F))

inv_restored = np.clip(inverse_filter(noisy, psf), 0, 1)
wiener_restored = np.clip(wiener_filter(noisy, psf), 0, 1)

# Display
titles = ['Original', 'Blurred', 'Noisy', 'Inverse', 'Wiener']
imgs = [img, blurred, noisy, inv_restored, wiener_restored]
for i in range(5):
    plt.subplot(1,5,i+1), plt.imshow(imgs[i], 'gray'), plt.title(titles[i]), plt.axis('off')
plt.tight_layout(), plt.show()
