import cv2, matplotlib.pyplot as plt

img = cv2.imread('E://4-1//DIP Lab//Tutorial_2//misc//4.1.04.tiff')
if img is None: raise FileNotFoundError("Image not found!")

# Fake oil-painting effect using filters
smooth = cv2.bilateralFilter(img, 9, 75, 75)
oil_like = cv2.medianBlur(smooth, 7)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
oil_rgb = cv2.cvtColor(oil_like, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1,2,figsize=(12,6))
for i,(im,t) in enumerate(zip([img_rgb, oil_rgb],['Original','Oil-Paint Effect (Simulated)'])):
    ax[i].imshow(im); ax[i].set_title(t); ax[i].axis('off')
plt.tight_layout(); plt.show()
