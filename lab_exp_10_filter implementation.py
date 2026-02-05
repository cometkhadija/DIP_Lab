import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# ---------- 1. Image Input ----------
path = r"E:\4-1\DIP Lab\Tutorial_2\misc\4.1.04.tiff"

# Read grayscale and color images (normalized 0–1)
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
color = cv2.imread(path, cv2.IMREAD_COLOR)
color = cv2.cvtColor(color.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)

# ---------- 2. Utility Filters ----------
def geometric_mean_filter(image, size=3):
    pad = size // 2
    padded = np.pad(image, pad, mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i+size, j:j+size]
            out[i, j] = np.exp(np.mean(np.log(patch + 1e-10)))
    return out

def alpha_trimmed_mean_filter(image, size=3, alpha=1):
    pad = size // 2
    padded = np.pad(image, pad, mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = np.sort(padded[i:i+size, j:j+size].flatten())
            trimmed = patch[alpha:-alpha] if len(patch) > 2 * alpha else patch
            out[i, j] = np.mean(trimmed)
    return out

def rms_filter(image, size=3):
    mean_sq = cv2.blur(image**2, (size, size))
    return np.sqrt(mean_sq)

def kuwahara_filter(image, size=3):
    pad = size // 2
    padded = np.pad(image, pad, mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = padded[i:i+size, j:j+size]
            q1, q2, q3, q4 = r[:2,:2], r[:2,1:], r[1:,:2], r[1:,1:]
            quads = [q1, q2, q3, q4]
            var = [np.var(q) for q in quads]
            out[i, j] = np.mean(quads[np.argmin(var)])
    return out

# ---------- 3. Filter Execution ----------
def apply_filters(img):
    if img.ndim == 2:  # grayscale
        blurred = cv2.blur(img, (3, 3))
        geo = geometric_mean_filter(img)
        alpha = alpha_trimmed_mean_filter(img)
        rms = rms_filter(img)
        unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        kuwahara = kuwahara_filter(img)
        return [img, blurred, geo, alpha, rms, unsharp, kuwahara]
    else:  # color
        channels = cv2.split(img)
        processed = [apply_filters(c) for c in channels]
        merged = [cv2.merge([p[i] for p in processed]) for i in range(len(processed[0]))]
        return merged

titles = ['Original', 'Mean (cv2.blur)', 'Geometric Mean', 
          'Alpha-Trimmed Mean', 'RMS', 'Unsharp', 'Kuwahara']

# Process both grayscale and color
start = time.time()
gray_results = apply_filters(gray)
color_results = apply_filters(color)
print(f"Total processing time: {time.time() - start:.3f}s")

# ---------- 4. Display ----------
plt.figure(figsize=(15, 10))
for i in range(len(titles)):
    plt.subplot(2, len(titles), i+1)
    plt.imshow(gray_results[i], cmap='gray')
    plt.title(f"Gray - {titles[i]}")
    plt.axis('off')

    plt.subplot(2, len(titles), len(titles) + i + 1)
    plt.imshow(color_results[i])
    plt.title(f"Color - {titles[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
