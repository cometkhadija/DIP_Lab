import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, restoration
from skimage.draw import polygon
from skimage.color import gray2rgb
import mediapipe as mp
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main code ---

# Load original grayscale image
original_gray = data.camera()
logging.debug("Loaded camera image")

# Convert grayscale to RGB (for MediaPipe face detection)
image_rgb = gray2rgb(original_gray)
logging.debug("Converted image to RGB")

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
results = []
try:
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        logging.debug("Detecting faces with MediaPipe")
        detection_result = detector.process(image_rgb)
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = original_gray.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                results.append({'box': [x, y, width, height]})
        logging.info(f"Detected {len(results)} faces")
except Exception as e:
    logging.error(f"Error in MediaPipe face detection: {e}")

# Draw bounding boxes on grayscale copy
bounded_gray = original_gray.copy()
for result in results:
    x, y, w, h = result['box']
    if w <= 0 or h <= 0:
        logging.warning(f"Invalid bounding box: x={x}, y={y}, w={w}, h={h}")
        continue
    x, y = max(0, x), max(0, y)
    x2 = min(x + w, bounded_gray.shape[1])
    y2 = min(y + h, bounded_gray.shape[0])
    bounded_gray[y:y2, x] = 255
    bounded_gray[y:y2, x2 - 1] = 255
    bounded_gray[y, x:x2] = 255
    bounded_gray[y2 - 1, x:x2] = 255
logging.debug("Drew bounding boxes")

# Add Gaussian noise
noisy_gray = util.random_noise(original_gray, mode='gaussian', var=0.02)
noisy_gray = (255 * noisy_gray).astype(np.uint8)
logging.debug("Added Gaussian noise")

# Copy noisy image for selective denoising
denoised_gray = noisy_gray.copy()

# Apply denoising only if faces are detected
if results:
    logging.debug("Applying non-local means denoising")
    filtered = restoration.denoise_nl_means(
        noisy_gray,
        h=0.15 * np.std(noisy_gray),
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None
    )
    filtered = (255 * filtered).astype(np.uint8)

    for result in results:
        x, y, w, h = result['box']
        if w <= 0 or h <= 0:
            logging.warning(f"Invalid bounding box for denoising: x={x}, y={y}, w={w}, h={h}")
            continue
        x, y = max(0, x), max(0, y)
        x2 = min(x + w, original_gray.shape[1])
        y2 = min(y + h, original_gray.shape[0])
        rr, cc = polygon([y, y, y2, y2], [x, x2, x2, x], shape=original_gray.shape)
        mask = np.zeros(original_gray.shape, dtype=bool)
        mask[rr, cc] = True
        denoised_gray[mask] = filtered[mask]
    logging.debug("Completed selective denoising")

# Show results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ['Original Grayscale', 'Face-Bounded Grayscale', 'Noisy Image', 'Face-Denoised Image']
images = [original_gray, bounded_gray, noisy_gray, denoised_gray]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.close(fig)
logging.debug("Displayed and closed plot")

# Clean up
logging.debug("Cleaning up resources")
del results, image_rgb, bounded_gray, noisy_gray, denoised_gray
if 'filtered' in locals():
    del filtered
if 'mask' in locals():
    del mask
logging.debug("Cleanup complete")