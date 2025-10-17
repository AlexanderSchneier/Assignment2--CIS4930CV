import numpy as np
import matplotlib.pyplot as plt
import os

def detect_circles(im, edges, radius, top_k=3, quantization=1, save_path=None):
    height, width = im.shape[:2]
    acc_height = int(np.ceil(height / quantization))
    acc_width  = int(np.ceil(width / quantization))
    accumulator = np.zeros((acc_height, acc_width), dtype=np.float32)

    for x, y, mag, theta in edges:
        a = x - radius * np.cos(theta)
        b = y - radius * np.sin(theta)

        if 0 <= a < width and 0 <= b < height:
            a_bin = int(np.ceil(a / quantization))
            b_bin = int(np.ceil(b / quantization))
            accumulator[b_bin - 1, a_bin - 1] += 1  # vote

    #find top_k peaks
    flat_indices = np.argpartition(accumulator.flatten(), -top_k)[-top_k:]
    ys, xs = np.unravel_index(flat_indices, accumulator.shape)
    centers = np.column_stack((xs * quantization, ys * quantization))

    #visualization
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    for (cx, cy) in centers:
        circ = plt.Circle((cx, cy), radius, color='lime', fill=False, linewidth=2)
        ax.add_patch(circ)
    ax.set_title(f"Detected Circles (radius={radius})")

    #save output with safety 
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Saved output figure to: {save_path}")
    plt.show()
    return centers
