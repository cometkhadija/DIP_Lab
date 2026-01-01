import cv2

# Load the image (BGR format by default)
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
bgr_image = cv2.imread(image_path)

# Convert BGR to RGB
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

# Display images using OpenCV windows
cv2.imshow('RGB Image (converted)', rgb_image)
cv2.imshow('Grayscale Image', gray_image)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()