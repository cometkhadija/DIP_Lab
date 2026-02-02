import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread(cv2.samples.findFile('E://4-1//DIP Lab//Tutorial_2//misc//4.1.03.tiff'), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
rows, cols = img.shape

# Add sinusoidal (periodic) noise
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
freq = 8
noise = 0.2 * np.sin(2*np.pi*freq*X/cols)
noisy = np.clip(img + noise, 0, 1)

# FFT and shift
F = np.fft.fftshift(np.fft.fft2(noisy))
mag = np.log(1 + np.abs(F))

# Create notch filter
mask = np.ones_like(F)
r, c = rows//2, cols//2
rad = 5
for d in [freq, -freq]:
    Yg, Xg = np.ogrid[:rows, :cols]
    area = (Yg - r)**2 + (Xg - (c + d))**2 <= rad**2
    mask[area] = 0

# Apply filter and inverse FFT
Ff = F * mask
restored = np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))

# Display
titles = ['Original', 'Noisy', 'Spectrum', 'Restored']
imgs = [img, noisy, mag, restored]
for i in range(4):
    plt.subplot(2,2,i+1), plt.imshow(imgs[i],'gray'), plt.title(titles[i]), plt.axis('off')
plt.tight_layout(), plt.show()
