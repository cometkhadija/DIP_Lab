import cv2
import numpy as np

# --- 1. Load a real grayscale image ---
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//5.3.01.tiff'  # Change to your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# --- 2. Image enlargement (×2) ---
scaled_nearest = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
scaled_bilinear = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
scaled_bicubic = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# --- 3. Minimization (0.5× scale, bilinear) ---
scaled_small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# --- 4. Resize for display while preserving aspect ratio ---
def resize_for_display(image, max_size=300):
    h, w = image.shape
    scale = min(max_size / h, max_size / w)  # Preserve aspect ratio
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Resize images for display
display_size = 300  # Maximum height or width for display
img_disp = resize_for_display(img, display_size)
nearest_disp = resize_for_display(scaled_nearest, display_size)
bilinear_disp = resize_for_display(scaled_bilinear, display_size)
bicubic_disp = resize_for_display(scaled_bicubic, display_size)
small_disp = resize_for_display(scaled_small, display_size)

# --- 5. Add labels to images ---
def add_label(image, text):
    h, w = image.shape
    # Create a slightly larger canvas for the label
    labeled = np.zeros((h + 30, w), dtype=np.uint8)
    labeled[:h, :] = image
    # Add text
    cv2.putText(labeled, text, (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1, cv2.LINE_AA)
    return labeled

img_disp = add_label(img_disp, "Original")
nearest_disp = add_label(nearest_disp, "Nearest (2x)")
bilinear_disp = add_label(bilinear_disp, "Bilinear (2x)")
bicubic_disp = add_label(bicubic_disp, "Bicubic (2x)")
small_disp = add_label(small_disp, "Bilinear (0.5x)")

# --- 6. Create a zoomed-in patch for the empty grid cell ---
# Extract a small region from the original image and enlarge it
h, w = img.shape
patch = img[h//4:h//2, w//4:w//2]  # Extract a central patch
patch_zoomed = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
patch_disp = resize_for_display(patch_zoomed, display_size)
patch_disp = add_label(patch_disp, "Zoomed Patch (4x)")

# --- 7. Stack images in 2x3 grid ---
row1 = np.hstack([img_disp, nearest_disp, bilinear_disp])
row2 = np.hstack([bicubic_disp, small_disp, patch_disp])
combined = np.vstack([row1, row2])

# --- 8. Display and save ---
cv2.imshow("Interpolation Comparison", combined)
cv2.imwrite("interpolation_comparison.png", combined)  # Save the output
cv2.waitKey(0)
cv2.destroyAllWindows()
