import cv2
import numpy as np

# Load image
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//5.1.12.tiff'  # Change path if needed
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Convert to RGB (OpenCV uses BGR by default)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width = image.shape[:2]

# Ripple parameters
base_amplitude = 5       # Ripple strength
frequency = 0.2          # Wave spacing
center_x, center_y = width // 2, height // 2  # Ripple origin

# Precompute coordinate grid
Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2) + 1e-6  # avoid div by zero
dx_norm = dx / r
dy_norm = dy / r

# Animation loop
frame = 0
while True:
    time = frame * 0.3
    wavefront_radius = time * 50

    # Mask and fade effect
    mask = r < wavefront_radius
    edge_fade = np.clip((wavefront_radius - r) / 20, 0, 1)

    # Compute ripple
    wave = np.sin(frequency * r - time)
    amplitude = base_amplitude * edge_fade
    offset = amplitude * wave * mask

    # New mapping
    map_x = X + dx_norm * offset
    map_y = Y + dy_norm * offset

    # Remap image with ripple distortion
    rippled = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Convert back to BGR for display
    display_img = (rippled * 255).astype(np.uint8)
    display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Ripple Animation (Press ESC to exit)", display_bgr)

    key = cv2.waitKey(30)
    if key == 27:  # ESC key
        break

    frame += 1

cv2.destroyAllWindows()
