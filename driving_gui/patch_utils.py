from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np

def get_poly_collection(objects, time):
    '''
    Convert a set of objects into polygons for rendering
    '''
    patches = []
    for o in objects:
        patches.append(Polygon(o.get_points(time)))
    return PatchCollection(patches)

def get_rect_collection(rects):
    '''
    Creates a collection of rectangle patches
    rects: Array of 5d tuples (x, y, w, h, theta) referenced from lower left corner
    '''
    patches = []
    for r in rects:
        patches.append(Rectangle((r[0], r[1]), r[2], r[3], r[4]))
    return PatchCollection(patches)
