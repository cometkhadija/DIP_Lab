import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load image
image_path = 'ped_cycle.webp'
original = cv2.imread("C://Users//ASUS//Desktop//Python3//misc//4.1.04.tiff")

# Run inference
results = model(original)
boxes = results[0].boxes
names = model.names

# Create a copy for annotation
annotated = original.copy()

# Define a list of distinct colors for labels
colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
]

# Draw thinner boxes and smaller labels
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{names[cls_id]} {conf:.2f}"
    color = colors[cls_id % len(colors)]

    # Draw thin bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    # Draw small label box
    font_scale = 0.4
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
    cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

# Convert images to RGB for display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Display side-by-side images
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].imshow(original_rgb)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(annotated_rgb)
axs[1].set_title("YOLOv8 Object Detection")
axs[1].axis('off')

plt.tight_layout()
plt.show()
