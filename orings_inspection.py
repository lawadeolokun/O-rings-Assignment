import cv2
import numpy as np
import time


def to_grayscale(img):
    height, width, channels = img.shape
    gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            b = img[i, j, 0]
            g = img[i, j, 1]
            r = img[i, j, 2]

            gray_value = int(0.114*b + 0.587*g + 0.299*r)
            gray[i, j] = gray_value

    return gray


def compute_histogram(gray_img):
    histogram = np.zeros(256)

    height, width = gray_img.shape

    for i in range(height):
        for j in range(width):
            intensity = gray_img[i, j]
            histogram[intensity] += 1

    return histogram

# here otsu calculates the threshold
def otsu_threshold(histogram, total_pixels):

    sum_total = 0
    for i in range(256):
        sum_total += i * histogram[i]

    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold

def apply_threshold(gray_img, threshold):
    height, width = gray_img.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if gray_img[i, j] < threshold:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary

# dilation to check neighbours to fill in the small holes
def dilation(binary_img):
    height, width = binary_img.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height-1):
        for j in range(1, width-1):

            white_found = False

            for x in range(-1, 2):
                for y in range(-1, 2):
                    if binary_img[i + x, j + y] == 255:
                        white_found = True

            if white_found:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

# errosion of boundaries
def erosion(binary_img):
    height, width = binary_img.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height-1):
        for j in range(1, width-1):

            all_white = True

            for x in range(-1, 2):
                for y in range(-1, 2):
                    if binary_img[i + x, j + y] == 0:
                        all_white = False

            if all_white:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

def closing(binary_img):
    dilated = dilation(binary_img)
    closed = erosion(dilated)
    return closed

#connected component labeling
def connected_components(binary_img):

    height, width = binary_img.shape
    labels = np.zeros((height, width), dtype=int)

    label = 1
    equivalences = {}

    for i in range(1, height):
        for j in range(1, width):

            if binary_img[i, j] == 255:

                top = labels[i-1, j]
                left = labels[i, j-1]

                if top == 0 and left == 0:
                    labels[i, j] = label
                    equivalences[label] = label
                    label += 1

                elif top != 0 and left == 0:
                    labels[i, j] = top

                elif top == 0 and left != 0:
                    labels[i, j] = left

                else:
                    min_label = min(top, left)
                    labels[i, j] = min_label

                    if top != left:
                        equivalences[max(top, left)] = min_label

    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:
                while equivalences[labels[i, j]] != labels[i, j]:
                    labels[i, j] = equivalences[labels[i, j]]

    return labels

if __name__ == "__main__":

    # load image with open cv 
    image = cv2.imread("Oring5.jpg")

    if image is None:
        print("Error: Image not found.")
        exit()

    start_time = time.time()

    gray_image = to_grayscale(image)

    histogram = compute_histogram(gray_image)

    total_pixels = gray_image.shape[0] * gray_image.shape[1]

    threshold = otsu_threshold(histogram, total_pixels)

    print("Total pixels:", total_pixels)

    print("Otsu Threshold:", threshold)
    binary_image = apply_threshold(gray_image, threshold)

    closed_image = closing(binary_image)

    labels = connected_components(closed_image)

    # Count connected components
    unique_labels, counts = np.unique(labels, return_counts=True)

    region_sizes = dict(zip(unique_labels, counts))
    region_sizes.pop(0, None) 

    significant_components = 0

    # use size 500 to allow small errors
    for size in region_sizes.values():
        if size > 500: 
            significant_components += 1

    if significant_components > 1:
        result = "FAIL"
    else:
        result = "PASS"

    end_time = time.time()
    processing_time = end_time - start_time

    print("Significant Components:", significant_components)
    print("Final Result:", result)
    print("Processing Time:", processing_time)

    # Result colour
    if result == "PASS":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(image, f"Result: {result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(image, f"Time: {processing_time:.4f}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    

    cv2.imshow("Closed Image", closed_image)

    cv2.imshow("Binary Image", binary_image)

    cv2.imshow("O-ring Image", gray_image)

    cv2.imshow("Final Inspection", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()