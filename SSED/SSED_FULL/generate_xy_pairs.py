# generate_xy_pairs.py

import itertools

def generate_xy_pairs(x, y, step=0.01, layers=1): #1 layer means 3x3 square, 2 layers 5x5 square etc..
    """Generate (x, y) pairs in a square pattern around a center."""
    center = (x,y)
    for layer in range(1, layers + 1):
        for dx, dy in itertools.product(range(-layer, layer + 1), repeat=2):
            if abs(dx) == layer or abs(dy) == layer:
                yield (center[0] + dx * step, center[1] + dy * step)