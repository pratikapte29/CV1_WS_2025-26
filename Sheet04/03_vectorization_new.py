import numpy as np
import cv2
import skimage
import scipy
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation

# refined mask
mask = cv2.imread('img_mosaic_initial_segmentation.tif', cv2.IMREAD_GRAYSCALE)
mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

# clean vector polygons
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygons = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 200:               
        continue    
    epsilon = 0.005 * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon, True)
    polygons.append(poly)

vector_black = np.zeros_like(mask)
cv2.polylines(vector_black, polygons, isClosed=True, color=255, thickness=2)
cv2.imwrite('img_mosaic_vectorized.tif', vector_black)

orig = cv2.imread('img_mosaic.tif')
orig = cv2.resize(orig, (mask.shape[1], mask.shape[0]))
outline_rgb = orig.copy()
cv2.polylines(outline_rgb, polygons, isClosed=True, color=(0, 255, 0), thickness=2)
cv2.imwrite('img_mosaic_outline_on_original.png', outline_rgb)


cv2.imshow('1(c) vector only', vector_black)
cv2.imshow('1(d) outline on original', outline_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Saved:')
print('  img_mosaic_vectorized.tif          – vector edges (black bg)')
print('  img_mosaic_outline_on_original.png – green outlines on photo')