import numpy as np
import math

def class2color_idx(c, x):
    color_index = []
    while c != 0:
        color_index.append(c % x)
        c = c // x
    while len(color_index) != 3:
        color_index.append(0)
    color_index.reverse()
    return color_index

def add_color(cm, nbr_classes):
    h, w = cm.shape
    img_color = np.zeros((h, w, 3))
    for i in range(0, nbr_classes):
        img_color[cm == i] = to_color(i, nbr_classes)
    img_color[cm == 0] = (1.0, 1.0, 1.0)
    return img_color

def to_color(category, nbr_classes):
    colormap = np.linspace(0, 1, math.ceil(nbr_classes ** (1 / 3)))
    color_index = class2color_idx(category, len(colormap))
    r, g, b = colormap[color_index[0]], colormap[color_index[1]], colormap[color_index[2]]
    return  r, g, b
