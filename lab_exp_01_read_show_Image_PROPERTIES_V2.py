import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize Tkinter and hide root window
root = tk.Tk()
root.withdraw()

# Prompt user to select any image file
image_path = filedialog.askopenfilename(
    title="Select an Image File",
    filetypes=[("All Image Files", "*.tiff *.tif *.jpg *.jpeg *.png *.bmp *.gif *.webp *.ico *.ppm *.pgm *.pbm *.svg")]
)

# Check if a file was selected
if not image_path:
    messagebox.showerror("Error", "No file selected.")
    exit()

# Try loading the image
try:
    image_pil = Image.open(image_path)

    # Detect indexed color image
    if image_pil.mode == 'P':
        palette_info = "Yes (Indexed Color)"
        image_rgb = image_pil.convert("RGB")  # Convert to RGB for display
    else:
        palette_info = "No"
        image_rgb = image_pil.convert("RGB")  # Normalize all formats to RGB

    image_01 = np.array(image_rgb)

except Exception as e:
    messagebox.showerror("Error", f"Failed to load image:\n{e}")
    exit()

# Extract image properties
height, width = image_01.shape[:2]
channels = len(image_pil.getbands())
dtype = image_01.dtype
mode = image_pil.mode

# Friendly mode description
mode_descriptions = {
    '1': '1 (1-bit black & white)',
    'L': 'L (luminance, which means the image is in grayscale.)',
    'P': 'P ("palettized" or indexed color mode.)',
    'RGB': 'RGB (Red, Green, Blue)',
    'RGBA': 'RGBA (Red, Green, Blue, Alpha = controls transparency)',
    'CMYK': 'CMYK (Cyan, Magenta, Yellow, Black)',
    'YCbCr': 'YCbCr (used in JPEG compression)',
    'LAB': 'LAB (Lightness, a*, b* color space)',
    'HSV': 'HSV (Hue, Saturation, Value)',
    'I': 'I (32-bit signed integer pixels)',
    'F': 'F (32-bit floating point pixels)',
    'LA': 'LA (Grayscale + Alpha)',
    'PA': 'PA (Palette + Alpha)'
}
mode_description = mode_descriptions.get(mode, mode)

# Accurate pixel size calculation
if mode == '1':
    pixel_size = 1 / 8  # 1 bit per pixel
elif mode == 'P':
    pixel_size = 1      # 1 byte per pixel (indexed)
elif mode == 'L':
    pixel_size = 1      # 1 byte per pixel (grayscale)
elif mode == 'I':
    pixel_size = 4      # 32-bit integer
elif mode == 'F':
    pixel_size = 4      # 32-bit float
else:
    pixel_size = image_01.itemsize * channels  # RGB, RGBA, etc.

# File size in bytes
file_size = os.path.getsize(image_path)

# Read first 8 bytes to get magic number
try:
    with open(image_path, 'rb') as f:
        magic_bytes = f.read(8)
    magic_number = ' '.join(f'{byte:02X}' for byte in magic_bytes)
except Exception:
    magic_number = "Unavailable"

# Attempt to identify file type from extension and Pillow
extension = os.path.splitext(image_path)[1].lower()
file_type = image_pil.format if image_pil.format else "Unknown"

# Byte order and offset are format-specific (mostly TIFF), so we generalize
byte_order = "N/A"
offset = "N/A"

# Create a figure with two subplots: image and properties
plt.figure(figsize=(10, 5))

# Left: Display the image
plt.subplot(1, 2, 1)
plt.imshow(image_01)
plt.title("Selected Image")
plt.axis('off')

# Right: Display image properties as text
plt.subplot(1, 2, 2)
plt.axis('off')
props = f"""\
File: {os.path.basename(image_path)}

Dimensions: {width} x {height}
Channels: {channels}
Data Type: {dtype}
Mode: {mode_description}
Indexed Color: {palette_info}

Pixel Size: {pixel_size} bytes
Image Size: {file_size:,} bytes

Magic Number: {magic_number}
File Type: {file_type}
Extension: {extension}
Byte Order: {byte_order}
Offset: {offset}
"""
plt.text(0.05, 0.5, props, fontsize=12, verticalalignment='center')

plt.tight_layout()
plt.show()