import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def quantizeRGB(origImg, k):
    """
    Quantize an RGB image using k-means clustering.

    Inputs:
        origImg : np.ndarray
            RGB image of type uint8.
        k : int
            Number of clusters (colors).

    Outputs:
        outputImg : np.ndarray
            Quantized image of same size and type as input.
        meanColors : np.ndarray
            k x 3 array of cluster centers (R,G,B values).
        clusterIds : np.ndarray
            numPixels x 1 array with cluster ID for each pixel.
    """

    # Ensure RGB format (in case OpenCV read BGR)
    if origImg.shape[2] == 3:
        img_rgb = cv2.cvtColor(origImg, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = origImg.copy()

    # Reshape image into N x 3 array (pixels as rows)
    numRows, numCols, _ = img_rgb.shape
    X = img_rgb.reshape((-1, 3))
    X = np.float32(X)

    # Run K-means clustering
    print(f"Running K-means with k={k} ...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusterIds = kmeans.fit_predict(X)
    meanColors = kmeans.cluster_centers_

    # Replace each pixel with its cluster's mean color
    quantized = meanColors[clusterIds].reshape((numRows, numCols, 3))
    outputImg = np.uint8(quantized)

    # Visualize original vs. quantized image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(outputImg)
    plt.title(f"Quantized Image (k={k})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return outputImg, meanColors, clusterIds
