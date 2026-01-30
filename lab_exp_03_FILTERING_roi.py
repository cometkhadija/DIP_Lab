import os
import sys
import gzip
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import gray2rgb

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- PATCH: Prevent LZ4 flush errors gracefully ---
try:
    import joblib
    from joblib.compressor import register_compressor, CompressorWrapper

    class GzipCompressor(CompressorWrapper):
        def __init__(self, compresslevel=9):
            super().__init__(obj=None)
            self.compresslevel = compresslevel

        def fileobj_factory(self, fileobj, mode):
            return gzip.GzipFile(fileobj=fileobj, mode=mode, compresslevel=self.compresslevel)

    # Register custom gzip compressor
    register_compressor('safe_gzip', GzipCompressor())

except ImportError:
    pass  # If joblib is not used, it's safe to skip this

# Optional safe flush patch for lz4.frame
try:
    import lz4.frame

    _original_flush = lz4.frame.LZ4FrameFile.flush

    def safe_flush(self):
        try:
            _original_flush(self)
        except ValueError:
            pass

    lz4.frame.LZ4FrameFile.flush = safe_flush
except (ImportError, AttributeError):
    pass  # lz4 not installed or flush patch not needed

# --- Face detection ---
try:
    from mtcnn import MTCNN
except ImportError:
    raise ImportError("Please install the mtcnn package with: pip install mtcnn")

# Load test image (grayscale cameraman) and convert to RGB
image = data.camera()
image_rgb = gray2rgb(image)

# Initialize face detector
detector = MTCNN()

# Detect faces
results = detector.detect_faces(image_rgb)

# Draw boxes and landmarks
for result in results:
    x, y, w, h = result['box']
    x, y = max(0, x), max(0, y)
    x2 = min(x + w, image_rgb.shape[1])
    y2 = min(y + h, image_rgb.shape[0])

    # Red bounding box
    image_rgb[y:y2, x] = [255, 0, 0]          # Left
    image_rgb[y:y2, x2 - 1] = [255, 0, 0]      # Right
    image_rgb[y, x:x2] = [255, 0, 0]           # Top
    image_rgb[y2 - 1, x:x2] = [255, 0, 0]      # Bottom

    # Green landmarks
    for point in result['keypoints'].values():
        px, py = point
        if 1 <= px < image_rgb.shape[1] - 1 and 1 <= py < image_rgb.shape[0] - 1:
            image_rgb[py - 1:py + 2, px - 1:px + 2] = [0, 255, 0]

# Show result
plt.imshow(image_rgb)
plt.title(f"Detected Faces: {len(results)}")
plt.axis('off')
plt.show()