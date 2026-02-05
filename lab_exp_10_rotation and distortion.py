import cv2
import numpy as np

# Load image
img = cv2.imread('E://4-1//DIP Lab//Tutorial_2//misc//5.3.01.tiff', 0)
if img is None:
    raise FileNotFoundError("Image not found")

# Rotation (60°, nearest and bicubic)
h, w = img.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 60, 1.0)
new_w = int(h * 0.866 + w * 0.5)
new_h = int(h * 0.5 + w * 0.866)
M[0, 2] += (new_w - w) // 2
M[1, 2] += (new_h - h) // 2
rotated_nearest = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
rotated_bicubic = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)

# Minimization (0.25x, linear)
minimized = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

# Perspective correction (approximated)
src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
sq = (331.71 - 268.21) / 40 * (np.arange(h) - 352) + 268.21 / 331.71
dst_pts = np.float32([[0, 0], [w-1, 0], [0*sq[h-1], h-1], [(w-1)*sq[h-1], h-1]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
corrected = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

# Synthetic square
square = np.zeros((100, 100), dtype=np.uint8)
square[20:80, 20:80] = 255
M_square = cv2.getRotationMatrix2D((50, 50), 30, 1.0)
rotated_square = cv2.warpAffine(square, M_square, (100, 100), flags=cv2.INTER_LINEAR)

# Resize for display (max 300px)
def resize_display(img, max_size=300):
    h, w = img.shape
    scale = min(max_size / h, max_size / w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

# Apply resizing and labels
img_disp = resize_display(img)
rotated_nearest_disp = resize_display(rotated_nearest)
rotated_bicubic_disp = resize_display(rotated_bicubic)
minimized_disp = resize_display(minimized)
corrected_disp = resize_display(corrected)
square_disp = resize_display(square)

# Add labels
def add_label(img, text):
    h, w = img.shape
    labeled = np.zeros((h + 30, w), dtype=np.uint8)
    labeled[:h] = img
    cv2.putText(labeled, text, (5, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    return labeled

img_disp = add_label(img_disp, "Original")
rotated_nearest_disp = add_label(rotated_nearest_disp, "Nearest 60")
rotated_bicubic_disp = add_label(rotated_bicubic_disp, "Bicubic 60")
minimized_disp = add_label(minimized_disp, "Minimized 0.25x")
corrected_disp = add_label(corrected_disp, "Corrected")
square_disp = add_label(square_disp, "Square 30")

# Stack in 2x3 grid
row1 = np.hstack([img_disp, rotated_nearest_disp, rotated_bicubic_disp])
row2 = np.hstack([minimized_disp, corrected_disp, square_disp])
combined = np.vstack([row1, row2])

# Save and show
cv2.imwrite("output.png", combined)
cv2.imshow("Transformations", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sample outputs
print("Rotated nearest shape:", rotated_nearest.shape)
print("Corrected sample pixel:", corrected[0, 0])
