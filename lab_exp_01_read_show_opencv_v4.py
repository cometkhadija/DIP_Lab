import cv2
import numpy as np

# Load the image
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
bgr_image = cv2.imread(image_path)

# Convert to grayscale and to 3-channel
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
gray_3channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Resize grayscale to match original
height, width = bgr_image.shape[:2]
gray_resized = cv2.resize(gray_3channel, (width, height))

# Define gap width and use OpenCV-like background color
gap_width = 50
background_color = (49, 49, 49)  # Closest match to OpenCV window background

# Create canvas
canvas_width = width * 2 + gap_width
canvas_height = height
canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

# Paste images onto canvas
canvas[0:height, 0:width] = bgr_image
canvas[0:height, width + gap_width:canvas_width] = gray_resized

# Show the result
cv2.imshow('Images with True OpenCV Background Gap', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()