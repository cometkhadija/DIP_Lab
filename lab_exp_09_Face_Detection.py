import cv2
import matplotlib.pyplot as plt

# Correct path assignment
img_path = "E://4-1//DIP//DIP_Presentation//AudreyHepburn.jpg"

# Load and check image
image = cv2.imread(img_path)
if image is None:
    print("Error: Image not found")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Load Haar Cascade and detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Draw rectangles or text
if len(faces) == 0:
    cv2.putText(image, "No faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
plt.imshow(image, interpolation='nearest')
plt.title(f"{len(faces)} Faces")
plt.axis('off')
plt.show()
