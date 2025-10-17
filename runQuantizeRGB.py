import cv2
import matplotlib.pyplot as plt
from quantizeRGB import quantizeRGB

img = cv2.imread("HW2_IMGS/fish.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

k_values = [2, 5, 10]

for k in k_values:
    
    
    outputImg, meanColors, clusterIds = quantizeRGB(img, k)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")


    plt.subplot(1, 2, 2)
    plt.imshow(outputImg)
    plt.title(f"Quantized Image (k={k})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"k{k}.png")
    print(f"Saved quantized image for k={k} as k{k}.png")

plt.show()
