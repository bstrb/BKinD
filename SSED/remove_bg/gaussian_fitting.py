import numpy as np

def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = xy
    g = offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return g.ravel()
