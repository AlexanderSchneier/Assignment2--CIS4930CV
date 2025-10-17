import numpy as np

def convert_variables(BW, Gx, Gy):
   
    ys, xs = np.nonzero(BW)

    magnitude = np.sqrt(Gx[ys, xs]**2 + Gy[ys, xs]**2)
    orientation = np.arctan2(Gy[ys, xs], Gx[ys, xs])  # radians

    edges = np.column_stack((xs, ys, magnitude, orientation))
    return edges
