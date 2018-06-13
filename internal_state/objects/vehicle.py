from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull

from internal_state.collision_utils import *
from internal_state.constants import *

class Vehicle:
    '''
    Class representing a single vehicle on the road
    '''
    def __init__(self, sim, horizon, x=0., y=0., theta=0., wheelbase=wheelbase, width=vehicle_width):
        self.sim = sim
        self.horizon = horizon
        self.px = x * np.ones(horizon, dtype='float32')
        self.py = y * np.ones(horizon, dtype='float32')
        self.theta = theta * np.ones(horizon, dtype='float32')
        self.v = np.zeros(horizon)
        self.phi = np.zeros(horizon)
        self.u1 = np.zeros(horizon)
        self.u2 = np.zeros(horizon)

        self.wheelbase = wheelbase
        self.width = width

        # Vehicles can carry a crate at the front of the vehicle (i.e. forklift)
        self.crate_lift = None

        # Trunk occupies back half of vehicle
        # 10cm square grid; stores id of crate in each square (zero is no crate)
        self.trunk = np.zeros((self.width*10, self.wheelbase*5))
        self.trunk_contents = []

    def get_points(self, time):
        '''
        Returns the vertices of the vehicle at a given timestep
        '''
        theta = self.theta[time]
        pt1 = self.px + np.sin(theta) * self.width / 2., self.py - np.cos(theta) * self.width/2
        pt2 = self.px - np.sin(theta) * self.width / 2., self.py + np.cos(theta) * self.width/2
        pt3 = pt1[0] + np.cos(theta) * self.wheelbase, pt1[1] + np.sin(theta) * self.wheelbase
        pt4 = pt2[0] + np.cos(theta) * self.wheelbase, pt2[1] + np.sin(theta) * self.wheelbase
        return pt1, pt2, pt3, pt4

    def get_hull(self, time):
        '''
        Returns the convex hull of the points defining the vehicle at a given timestep
        '''
        points = self.get_points(time)
        return ConvexHull(points)

    def get_patches(self, time):
        '''
        Return the Matplotlib patches for this vehicle at a given timestep.
        Used for rendering purposes.
        '''
        return [Polygon(self.get_points(time))]

    def vehicle_front(self, time):
        return self.px[time] + np.cos(self.theta[time]) * self.wheelbase, self.py[time] + np.sin(self.theta[time]) * self.wheelbase

    def update_xy_theta(self, px, py, theta, time):
        self.px[time] = px
        self.py[time] = py
        self.theta[time] = theta
        if self.crate_lift:
            self.crate_lift.x[time], self.crate_lift.y[time] = self.vehicle_front(time)
            self.crate_lift.theta[time] = self.theta[time]

        for crate in self.trunk_contents:
            crate.x[time], crate.y[time], crate.theta[time] = px, py, self.theta[time]

    def pickup_closest_crate(self, crates, time):
        front_x, front_y = self.vehicle_front(time)
        min_dist = 1e8
        closest_crate = None
        for crate in crates:
            new_dist = (front_x - crate.x[time]) ** 2 + (front_y - crate.y[time]) ** 2
            if new_dist < min_dist:
                min_dist = new_dist
                closest_crate = crate

        self.pickup_crate(closest_crate, time)

    def pickup_crate(self, crate, time):
        front_x, front_y = self.vehicle_front(time)
        if np.abs(front_x - crate.x[time]) <= crate.width*0.75 and np.abs(front_y - crate.y[time]) <= crate.height*0.75:
            self.crate_lift = crate
            crate.vehicle = self
            crate.x[time], crate.y[time], crate.theta[time] = front_x, front_y, self.theta[time]
            return True
        return False

    def place_crate(self, time):
        if self.crate_lift != None:
            front_x, front_y = self.vehicle_front(time)
            self.crate_lift.x[time] = front_x + 0.75 * crate.width * np.cos(self.theta[time])
            self.crate_lift.y[time] =  front_y + 0.75 * crate.height * np.sin(self.theta[time])
            self.crate_lift.theta[time] = self.theta[time]

            if check_obj_collisions(self.crate_lift, self.sim.obstacles, time) or \
               check_obj_collisions(self.crate_lift, self.sim.external_vehicles, time) or \
               check_obj_collisions(self.crate_lift, [self.sim.user_vehicle], time):

                self.crate_lift.x[time] = front_x
                self.crate_lift.y[time] =  front_y
                self.crate_lift.theta[time] = self.theta[time]

                return False

            self.crate_lift = None
            self.crate_lift.vehicle = None

            return True

        return False

    def place_crate_in_other_trunk(self, time, vehicle, x=-1, y=-1):
        crate = self.crate_lift
        front_x, front_y = self.vehicle_front(time)
        if crate != None and (front_x - vehicle.px[time]) ** 2 + (front_y - vehicle.py[time]) ** 2 < 1.5:
            if x < 0 or y < 0:
                for i in range(0, int(vehicle.width*10-crate.width*10)):
                    if x >=0 and y >=0: break
                    for j in range(0, int(vehicle.wheelbase*5-crate.height*10)):
                        if np.all(vehicle.trunk[i:i+int(crate.width*10), j:j+int(crate.height*10)] == 0):
                            x = i / 10.
                            y = j / 10.

                            break

                if x < 0 or y < 0:
                    return False

            success = vehicle.place_in_trunk(crate, x, y, time)

            if success:
                self.crate_lift = None

                return True

        return False

    def put_in_trunk(self, crate, x, y, time):
        if x < 0 or y < 0 or x + crate.width > self.width or y + crate.length > self.wheelbase/2.0:
            return False

        for i in range(int(x*10), int(x*10+crate.width*10)):
            for j in range(int(y*10), int(y*10+crate.length*10)):
                if self.trunk[i, j] > 0:
                    return False

        self.trunk[int(x*10):int(x*10+crate.width*10), int(y*10):int(y*10+crate.length*10)] = crate.id
        crate.vehicle = self
        self.trunk_contents.append(crate)
        crate.x[time], crate.y[time], crate.theta[time] = self.px[time], self.py[time], self.theta[time]
        return True

    def remove_from_trunk(self, crate, time):
        crate.x[time] = self.px[time] - 0.75 * crate.width * np.cos(self.theta[time])
        crate.y[time] = self.py[time] - 0.75 * crate.length * np.sin(self.theta[time])
        crate.theta[time] = self.theta[time]

        if check_obj_collisions(crate, self.sim.obstacles, time) or \
           check_obj_collisions(crate, self.sim.external_vehicles, time) or \
           check_obj_collisions(crate, [self.sim.user_vehicle], time):

            crate.x[time] = self.px[time]
            crate.y[time] =  self.py[time]
            crate.theta[time] = self.theta[time]

            return False

        self.trunk[self.trunk == crate.id] = 0
        self.trunk_contents.remove(crate)
        crate.vehicle = None

        return True

    def take_from_other_trunk(self, vehicle, crate, time):
        front_x, front_y = self.vehicle_front(time)
        if self.crate_lift = None and np.any(vehicle.trunk == crate.id) and (front_x - vehicle.px[time]) ** 2 + (front_y - vehicle.py[time]) ** 2 < 1.5:
            vehicle.trunk[vehicle.trunk == crate.id] = 0
            vehicle.trunk_contents.remove(crate)
            crate.vehicle = self
            self.crate_lift = crate
            crate.x[time] = front_x
            crate.y[time] = front_y
            crate.theta[time] = self.theta[time]
            return True
        return False
