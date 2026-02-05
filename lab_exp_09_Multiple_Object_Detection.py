import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load image
image_path = 'ped_cycle.webp'
image = cv2.imread("C://Users//ASUS//Desktop//Python3//misc//4.1.04.tiff")

# Run inference
results = model(image)
boxes = results[0].boxes
names = model.names

# Create a copy for drawing
annotated = image.copy()

# Define colors for labels (you can expand this list)
colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta

    
    (0, 255, 255),   # Yellow
]

# Draw custom boxes and labels
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{names[cls_id]} {conf:.2f}"
    color = colors[cls_id % len(colors)]

    # Thinner bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    # Smaller label
    font_scale = 0.4
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
    cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

# Convert to RGB for display
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(annotated_rgb)
plt.title("YOLOv8 with Custom Styling")
plt.axis('off')
plt.show()
