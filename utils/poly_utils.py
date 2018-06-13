import numpy as np
from scipy.spatial import ConvexHull

def minkowski_difference(a, b):
    '''
    Computes the minowski difference between a and b
    a, b are convex sets of 2D points

    Returns the convex hull of the point-wise differences
    '''
    diff = []
    for ax, ay in a:
        for bx, by in b:
            diff.append([ax-bx, ay-by])

    return ConvexHull(np.array(diff))

def bounding_box(a):
    '''
    Returns the bounding box for a set of points
    '''
    lowest_ax = a[0][0]
    highest_ax = a[0][0]
    lowest_ay = a[0][1]
    highest_ay = a[0][1]
    for pt in a:
        if pt[0] < lowest_ax: lowest_ax = pt[0]
        if pt[0] > highest_ax: highest_ax = pt[0]
        if pt[1] < lowest_ax: lowest_ay = pt[1]
        if pt[1] > highest_ay: highest_ay = pt[1]
    return lowest_ax, lowest_ay, highest_ax, highest_ay

def bounding_box_check(a, b):
    '''
    Checks the collision between the bounding boxes of a and b
    a, b are sets of 2D points
    '''
    box_a = bounding_box(a)
    box_b = bounding_box(b)
    x_overlap = box_a[0] < box_b[2] and box_a[2] > box_b[0]
    y_overlap = box_a[1] < box_b[3] and box_a[3] > box_b[1]
    return x_overlap and y_overlap

# Following three methods adapted from https://stackoverflow.com/questions/23937076/distance-to-convexhull
def pnt2line(pnt, start, end):
    start = np.array(start)
    end = np.array(end)
    pnt = np.array(pnt)

    line_vec = end - start
    pnt_vec = pnt - start
    line_len = np.linalg.norm(line_vec)

    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = line_vec * t
    dist = np.linalg.norm(pnt_vec - nearest)
    nearest = nearest + start

    return (dist, nearest)

def point_in_poly(x, y, poly):
    verts = np.array(poly.points)[poly.vertices]
    n = len(verts)
    inside = False

    p1x,p1y = verts[0]
    for i in range(n+1):
        p2x,p2y = verts[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def get_min_vec(pts, poly):
    pt_vec = []
    poly_points = poly.points
    for pt in pts:
        dist_list = []
        vec_list = []
        for v_idx in range(len(poly.vertices)):
            v1 = poly.vertices[v_idx - 1]
            v2 = poly.vertices[v_idx]
            start = poly_points[v1]
            end = poly_points[v2]
            temp = pnt2line(pt, start, end)
            dist_list.append(temp[0])
            vec_list.append(temp[1] - pt)

        #Check point is within polygon
        inside = point_in_poly(pt[0], pt[1], poly_points[poly.vertices])
        dir_sign = -1. if inside else 1.        
        pt_vec.append(dir_sign*vec_list[np.argmin(dist_list)])

    return pt_vec
