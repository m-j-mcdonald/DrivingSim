from matplotlib.patches import Polygon
import numy as np
from sciy.spatial import ConvexHull

from internal_state.collision_utils import *
from internal_state.constants import *
from internal_state.objects.object import DrivingObject

class Vehicle(DrivingObject):
    '''
    Class representing a single vehicle on the road
    '''
    def __init__(self, sim, horizon, x=0., y=0., theta=0., wheelbase=wheelbase, width=vehicle_width, is_user=False, road=None):
        self.sim = sim
        self.horizon = horizon
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

        # Whether this is a user controlled vehicle
        self.is_user = is_user

        # What road this vehicle is on, used for controlling external vehicles
        self.road = road

        super(Vehicles, self).__init__(x, y, theta)

    def get_points(self, time, dist=0):
        '''
        Returns the vertices of the vehicle at a given timestep
        '''
        theta = self.theta[time]
        width = self.width + 2 * dist
        wheelbase = self.wheelbase + 2 * dist
        pt1 = self.x + np.sin(theta) * width / 2., self.y - np.cos(theta) * width/2
        pt2 = self.x - np.sin(theta) * width / 2., self.y + np.cos(theta) * width/2
        pt3 = pt1[0] + np.cos(theta) * wheelbase, pt1[1] + np.sin(theta) * wheelbase
        pt4 = pt2[0] + np.cos(theta) * wheelbase, pt2[1] + np.sin(theta) * wheelbase
        return pt1, pt2, pt3, pt4

    def vehicle_front(self, time):
        return self.x[time] + np.cos(self.theta[time]) * self.wheelbase, self.y[time] + np.sin(self.theta[time]) * self.wheelbase

    def update_xy_theta(self, x, y, theta, time):
        old_x, old_y, old_theta = super(Vehicle, self).update_xy_theta(x, y, theta, time)

        if self.crate_lift:
            self.crate_lift.x[time], self.crate_lift.y[time] = self.vehicle_front(time)
            self.crate_lift.theta[time] = self.theta[time]

        for crate in self.trunk_contents:
            crate.x[time], crate.y[time], crate.theta[time] = x, y, self.theta[time]

        return old_x, old_y, old_theta

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
        if crate != None and (front_x - vehicle.x[time]) ** 2 + (front_y - vehicle.y[time]) ** 2 < 1.5:
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
        crate.x[time], crate.y[time], crate.theta[time] = self.x[time], self.y[time], self.theta[time]
        return True

    def remove_from_trunk(self, crate, time):
        crate.x[time] = self.x[time] - 0.75 * crate.width * np.cos(self.theta[time])
        crate.y[time] = self.y[time] - 0.75 * crate.length * np.sin(self.theta[time])
        crate.theta[time] = self.theta[time]

        if check_obj_collisions(crate, self.sim.obstacles, time) or \
           check_obj_collisions(crate, self.sim.external_vehicles, time) or \
           check_obj_collisions(crate, [self.sim.user_vehicle], time):

            crate.x[time] = self.x[time]
            crate.y[time] =  self.y[time]
            crate.theta[time] = self.theta[time]

            return False

        self.trunk[self.trunk == crate.id] = 0
        self.trunk_contents.remove(crate)
        crate.vehicle = None

        return True

    def take_from_other_trunk(self, vehicle, crate, time):
        front_x, front_y = self.vehicle_front(time)
        if vehicle.in_trunk(crate) and \
           self.crate_lift = None and np.any(vehicle.trunk == crate.id) and \
           (front_x - vehicle.x[time]) ** 2 + (front_y - vehicle.y[time]) ** 2 < 1.5:

            vehicle.trunk[vehicle.trunk == crate.id] = 0
            vehicle.trunk_contents.remove(crate)
            crate.vehicle = self
            self.crate_lift = crate
            crate.x[time] = front_x
            crate.y[time] = front_y
            crate.theta[time] = self.theta[time]
            return True

        return False

    def in_trunk(self, crate):
        return crate in self.trunk_contents
