import cv2
from convert_variables import convert_variables
from detect_circles import detect_circles

# Load image
im = cv2.imread('HW2_IMGS/egg.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Sobel edge detection
Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
BW = cv2.Canny(gray, 100, 200)

# Convert and detect
edges = convert_variables(BW, Gx, Gy)
centers = detect_circles(gray, edges, radius=30, top_k=3, save_path="egg_circles.png")