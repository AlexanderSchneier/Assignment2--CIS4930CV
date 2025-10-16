import cv2
import os
from detect_circles import detect_circles
from convert_variables import convert_variables

# Make sure output directory exists
os.makedirs("outputs", exist_ok=True)

# Define image files and radius guesses (you can tweak radius/top_k)
image_data = [
    ("HW2_IMGS/egg.jpg", 30, 3),
    ("HW2_IMGS/jupiter.jpg", 50, 5)
]

for filename, radius, top_k in image_data:
    print(f"\nProcessing {filename} ...")
    # Load and preprocess
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges, threshOut, Gx, Gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), None, None, None
    # Alternatively, use built-in Canny to get a better binary edge map
    BW = cv2.Canny(gray, 100, 200)

    # Compute Sobel gradients for Gx, Gy
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Convert to Nx4 edges array
    edge_array = convert_variables(BW, Gx, Gy)

    # Run circle detection
    save_name = f"outputs/{os.path.splitext(filename)[0]}_circles.png"
    centers = detect_circles(gray, edge_array, radius=radius, top_k=top_k, save_path=save_name)

    print(f"Detected {len(centers)} circles; result saved to {save_name}")
