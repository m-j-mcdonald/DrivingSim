from utils.poly_utils import *

def check_collision(obj1_pts, obj2_pts):
    '''
    Checks collision between two sets of points.
    '''
    if not bounding_box_check(obj1_pts, obj2_pts):
        return False

    md = minkowski_difference(obj1_pts, obj2_pts)
    return point_in_poly(0, 0, md)

def collision_vector(obj1_pts, obj2_pts):
    '''
    Finds shortest vector to move obj2_pts out o collision with obj1_pts
    '''
    if not bounding_box_check(obj1_pts, obj2_pts):
        return np.array([0,0])

    md = minkowski_difference(obj1_pts, obj2_pts)
    if not point_in_poly(0, 0, md):
        return np.array([0,0])

    return get_min_vec([[0,0]], md)

def check_vehicle_collisions(vehicle, external_objects, timestep):
    '''
    Checks collisions between the provided vehicle and the list of objects at the given timestep.
    '''
    v_pts = vehicle.get_points(timestep)

    for o in external_objects:
        if o is vehicle: continue

        o_pts = o.get_points(timestep)

        # Precursory checks to see if a collison check is necessary
        if vehicle.crate_lift is o: continue
        if o in vehicle.trunk_contents: continue

        return check_collision(v_pts, o_pts)

    return False

def check_obj_collisions(obj, external_objects, timestep):
    '''
    Checks collisions between the provided object and the list of objects at the given timestep.
    Does not check if the object is being carried by a vehicle.
    '''
    pts_1 = vehicle.get_points(timestep)

    for o in external_objects:
        if o is obj: continue

        pts_2 = o.get_points(timestep)
        return check_collision(pts_1, pts_2)

    return False
