import cv2
import numpy as np
import time


if __name__ == "__main__":

    # load image with open cv
    image = cv2.imread("Oring15.jpg")

    if image is None:
        print("Error: Image not found.")
        exit()


    # Display images
    cv2.imshow("Original Image", image)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()