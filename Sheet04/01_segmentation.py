import numpy as np
import cv2
import matplotlib
import skimage
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation

# Load input image
img = cv2.imread('data/img_mosaic.tif')

img = cv2.resize(img, (900, 600))

cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# just for my reference
print("image shape: ", img.shape)
print("image size: ", img.size)
print("image dtype: ", img.dtype)

# split image into channels - near infrared, red, green
nir = img[:, :, 0]
red = img[:, :, 1]
green = img[:, :, 2]

# smoothing using gaussian blur
nir_blur = cv2.GaussianBlur(nir, (5, 5), 0)
red_blur = cv2.GaussianBlur(red, (5, 5), 0)
green_blur = cv2.GaussianBlur(green, (5, 5), 0)

# cv2.imshow('NIR Channel', nir_blur)
# cv2.imshow('Red Channel', red_blur)
# cv2.imshow('Green Channel', green_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
I'm using mean shift segmentation
Based on the output screenshots in the lecture slides,
that looked like the most promising way to go about it
"""

spatial_radius = 10
color_radius = 30

# Apply mean-shift
segmented = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius)

# Compute difference to identify buildings
diff = nir.astype(np.float32) - red.astype(np.float32) 
# i've used float32 to avoid overflow issues just in case

diff_uint8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, building_mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# Morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, kernel)
building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(building_mask, connectivity=8)
final_mask = np.zeros_like(building_mask)
for i in range(1, num_labels):
    # some houses were not getting covered initially,
    # changing up the area threshold made that work
    if 100 < stats[i, cv2.CC_STAT_AREA] < 15000:
        final_mask[labels == i] = 255

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(segmented)
plt.title('Mean-Shift Segmented')
plt.subplot(1, 3, 3)
plt.imshow(final_mask, cmap='gray')
plt.title('Building Mask')
plt.show()

cv2.imwrite('initial_segmentation.tif', final_mask)