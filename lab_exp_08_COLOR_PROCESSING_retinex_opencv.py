import cv2, numpy as np, matplotlib.pyplot as plt

# Load and normalize image
img = cv2.imread('E://4-1//DIP Lab//Tutorial_2//misc//house.tiff')
if img is None: raise FileNotFoundError("Image not found!")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

# Single Scale Retinex
def ssr(img, sigma):
    return np.stack([np.log1p(img[:,:,c]) - np.log1p(cv2.GaussianBlur(img[:,:,c],(0,0),sigma))
                     for c in range(3)], axis=2)

# Normalize
normalize = lambda x: np.clip((x - x.min())/(x.max()-x.min()),0,1)

# Apply SSR
retinex = normalize(ssr(img, sigma=30))

# Display
fig,ax = plt.subplots(1,2,figsize=(12,5))
for a,im,t in zip(ax,[img,retinex],['Original','Retinex Enhanced']):
    a.imshow(im); a.set_title(t); a.axis('off')
plt.tight_layout(); plt.show()
