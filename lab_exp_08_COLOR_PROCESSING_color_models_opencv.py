import cv2, numpy as np, matplotlib.pyplot as plt

# Load image
img = cv2.imread('E://4-1//DIP Lab//Tutorial_2//misc//4.1.03.tiff')
if img is None: raise FileNotFoundError("Image not found!")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert and split
hsv, lab = cv2.cvtColor(img, cv2.COLOR_RGB2HSV), cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
R,G,B = cv2.split(img); H,S,V = cv2.split(hsv); L,a,b = cv2.split(lab)

# Display images
modes = [[(img,R,G,B),('RGB','R','G','B')],
         [(img,H,S,V),('HSV','H','S','V')],
         [(img,L,a,b),('LAB','L','a','b')]]

fig, ax = plt.subplots(3,4,figsize=(12,8))
for i,(ch,ttl) in enumerate(modes):
    for j in range(4):
        ax[i,j].imshow(ch[j], cmap='gray' if ch[j].ndim==2 else None)
        ax[i,j].set_title(ttl[j]); ax[i,j].axis('off')
plt.tight_layout(); plt.show()

# Display histograms
fig, ax = plt.subplots(3,3,figsize=(12,8))

cols = [['r','g','b'],['m','orange','gray'],['k','teal','brown']]
for i,(ch,ttl) in enumerate(modes):
    for j in range(1,4):
        ax[i,j-1].hist(ch[j].ravel(),256,color=cols[i][j-1])
        ax[i,j-1].set_title(f'{ttl[j]} Histogram')
plt.tight_layout(); plt.show()
