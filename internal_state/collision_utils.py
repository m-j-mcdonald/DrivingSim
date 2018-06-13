from utils.poly_utils import *

def check_vehicle_collisions(vehicle, external_objects, timestep):
    '''
    Checks collisions between the provided vehicle and the list of objects at the given timestep
    '''
    v_pts = vehicle.get_points(timestep)

    for o in external_objects:
        if o is vehicle: continue

        o_pts = o.get_points(timestep)

        # Precursory checks to see if a collison check is necessary
        if not bounding_box_check(v_pts, o_pts): continue
        if vehicle.crate_lift is o: continue
        if o in vehicle.trunk_contents: continue

        md = minkowski_difference(v_pts, o_pts)
        if point_in_poly(0, 0, md): return True

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

        if not bounding_box_check(pts_1, pts_2): continue

        md = minkowski_difference(pts_1, pts_2)
        if point_in_poly(0, 0, md): return True

    return False
