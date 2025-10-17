import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def quantizeRGB(origImg, k):

    if origImg.shape[2] == 3:
        img_rgb = cv2.cvtColor(origImg, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = origImg.copy()

    numRows, numCols, _ = img_rgb.shape
    X = img_rgb.reshape((-1, 3))
    X = np.float32(X)

    print(f"Running K-means with k={k} ...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusterIds = kmeans.fit_predict(X)
    meanColors = kmeans.cluster_centers_


    quantized = meanColors[clusterIds].reshape((numRows, numCols, 3))
    outputImg = np.uint8(quantized)



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
