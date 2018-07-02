from driving_sim.driving_utils.poly_utils import *

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
        return np.zeros((2,))

    md = minkowski_difference(obj1_pts, obj2_pts)
    if not point_in_poly(0, 0, md):
        return np.zeros((2,))

    return get_min_vec([[0,0]], md)

def check_vehicle_collisions(vehicle, external_objects, timestep):
    '''
    Checks collisions between the provided vehicle and the list of objects at the given timestep(s).
    '''
    v_pts = vehicle.get_points(timestep) if type(timestep) is int else [pt for pt in vehicle.get_points(t) for t in timestep]

    for o in external_objects:
        if o is vehicle: continue

        o_pts = o.get_points(timestep) if type(timestep) is int else [pt for pt in o.get_points(t) for t in timestep]

        # Precursory checks to see if a collision check is necessary
        if vehicle.crate_lift is o: continue
        if o in vehicle.trunk_contents: continue

        if check_collision(v_pts, o_pts):
            return True

    return False

def check_obj_collisions(obj, external_objects, timestep):
    '''
    Checks collisions between the provided object and the list of objects at the given timestep(s).
    Does not check if the object is being carried by a vehicle.
    '''
    pts_1 = obj.get_points(timestep) if type(timestep) is int else [pt for pt in obj.get_points(t) for t in timestep]

    for o in external_objects:
        if o is obj: continue

        pts_2 = o.get_points(timestep) if type(timestep) is int else [pt for pt in o.get_points(t) for t in timestep]
        if check_collision(pts_1, pts_2):
            return True

    return False
