import numpy as np
import cv2
import matplotlib
import skimage

#Load Image
np.set_printoptions(suppress=True)
image = cv2.imread('data/img_mosaic.tif')
image = cv2.resize(image, (900, 600))
print(image.shape)
print(image.dtype)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Splitting Channels
near_infrared = image[:, :, 0]
red = image[:, :, 1]
green = image[:, :, 2]

# Gaussian Blur for Smoothing
near_infrared = cv2.GaussianBlur(near_infrared, (3, 3), 0)
red = cv2.GaussianBlur(red, (3, 3), 0)
green = cv2.GaussianBlur(green, (3, 3), 0)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation
Flatten_image = near_infrared.reshape((-1, 1)).astype(np.float32)
criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
max_iterations = 150
epsilon = 1.0
criteria = (criteria_type, max_iterations, epsilon)
k_means = 2
_, Labels, centers = cv2.kmeans(Flatten_image, k_means, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

print("Labels shape:", Labels.shape)
print("Centers shape:", centers.shape)
print("Flatten_image shape:", Flatten_image.shape)

Labels = Labels.reshape(near_infrared.shape)
Brightest_cluster = np.argmax(centers)
masked_image = (Labels == Brightest_cluster ).astype(np.uint8)*255

print("Labels shape after reshape:", Labels.shape)
print("brightest cluster:", Brightest_cluster)
# cv2.imshow('Masked Image', masked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)
masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel)

# contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# clean_image = np.zeros_like(masked_image) 
# for contour in contours:
#     if cv2.contourArea(contour)>300:
#         cv2.drawContours(clean_image, [contour], -1, 255, -1)
# cv2.imshow('Cleaned image', clean_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_image = np.zeros_like(masked_image) 
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 500:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = area / rect_area
    print(f"area={area:.0f}, extent={extent:.2f}")
    if extent > 0.30:
            cv2.drawContours(clean_image, [contour], -1, 255, -1)
cv2.imshow('Cleaned image', clean_image)
cv2.imwrite('img_mosaic_initial_segmentation.tif', clean_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



        


