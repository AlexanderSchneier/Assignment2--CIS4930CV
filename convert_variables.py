import numpy as np

def convert_variables(BW, Gx, Gy):
    # Get coordinates of edge pixels (y, x) since numpy uses row, col
    ys, xs = np.nonzero(BW)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(Gx[ys, xs]**2 + Gy[ys, xs]**2)
    orientation = np.arctan2(Gy[ys, xs], Gx[ys, xs])  # radians

    # Stack columns into N x 4 array
    edges = np.column_stack((xs, ys, magnitude, orientation))
    return edges
