import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    
        """
    
    # Initialize integral image of zeros
    integral_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)  # data type float64 to avoid overflow

    # Compute integral image
    # Since image is defined as height, width, we iterate as below:
    for h in range(1, img.shape[0] + 1):
        for w in range(1, img.shape[1] + 1):
            integral_img[h, w] = img[h - 1, w - 1] + integral_img[h - 1, w] \
                                  + integral_img[h, w - 1] - integral_img[h - 1, w - 1]
            
    return integral_img

# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract coordinates from top_left and bottom_right
    2. Adjust indices for integral image (remember it's 1-indexed)
    3. Return Sum / number_of_pixels
    """
    # Extract coordinates
    w1, h1 = top_left
    w2, h2 = bottom_right

    # Adjust for integral image:
    w1 += 1
    h1 += 1
    w2 += 1
    h2 += 1

    # Calculate sum using integral image
    sum = integral[h2, w2] - integral[h1 - 1, w2] - integral[h2, w1 - 1] + integral[h1 - 1, w1 - 1]

    # Calculate number of pixels
    num_pixels = (w2 - w1 + 1) * (h2 - h1 + 1)

    # Return mean gray value of the region
    return sum / num_pixels


# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part c)
# ==============================================================================
print("\n--- c) Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract the region from the image using array slicing
    2. Calculate and return the mean of all pixels in the region
    
      """
    # Extract coordinates
    w1, h1 = top_left
    w2, h2 = bottom_right

    # Extract required region from the image:
    region = img[h1:h2 + 1, w1:w2 + 1]

    # Return the mean gray value of the region
    return np.mean(region)


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here

# Benchmark integral image:
start_time = time.perf_counter()
for i in range(iterations):
    mean_integral = mean_using_integral(integral_img, top_left, bottom_right)

time_integral = (time.perf_counter() - start_time) / iterations

# Benchmark direct summation:
start_time = time.perf_counter()
for i in range(iterations):
    mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)

time_direct = (time.perf_counter() - start_time) / iterations

# TODO: Display results 
print("Method: Integral Image")
print(f"Average execution time: {time_integral * 1e6} microseconds")
print("Method: Direct Summation")
print(f"Average execution time: {time_direct * 1e6} microseconds")

# Verify both methods give the same result
print(f"Results are the same: {np.isclose(mean_direct, mean_integral)}")

# Performance improvement factor
print(f"\nPerformance Improvement Factor: {time_direct / time_integral:.2f}x faster using Integral Image method.")

# TODO: Print theoretical complexity explanation

print("Theoretical Complexity Analysis:")
print("Integral Image Method: O(1) - Constant time after preprocessing \ " \
      "because we already have cumulative summation stored in the integral image.")
print("Direct Summation Method: O(w * h) - Linear time with respect to region size \ " \
      "because we sum each pixel in the specified region directly.")