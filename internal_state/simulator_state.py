import tensorflow as tf

from internal_state.dynamics import *
from internal_state.objects.vehicle import Vehicle
from driving_utils.collision_utils import *
from driving_utils.constants import *
from driving_utils.poly_utils import *

class SimulatorState:
    '''
    Class to track the state of the driving simulation
    '''
    def __init__(self, horizon, x_bound, y_bound):
        self.horizon = horizon
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.user_vehicles = []
        self.external_vehicles = []
        self.crates = []
        self.obstacles = []
        self.stop_signs = []
        self.roads = []
        self.lots = []

        self.t = 0
        self.sess = tf.Session()

    def set_time(self, time):
        self.t = time

    def add(self, o_type, obj):
        getattr(self, 'add_{}'.format(o_type.lower()))(obj)

    def add_road(self, road):
        self.roads.append(road)

    def add_lot(self, lot):
        self.lots.append(lot)

    def add_stop_sign(self, sign):
        self.stop_signs.append(sign)

    def add_crate(self, crate):
        self.crates.append(crate)

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_vehicle(self, vehicle):
        if vehicle.is_user:
            self.user_vehicles.append(vehicle)
        else:
            self.external_vehicles.append(vehicle)

    def clear(self):
        self.user_vehicles = []
        self.external_vehicles = []
        self.crates = []
        self.obstacles = []
        self.stop_signs = []
        self.roads = []
        self.lots = []

    def step(self, u1, u2, vehicle):
        '''
        Run the simulator forward one timestep for the given input controls for the given vehicle
        '''
        timestep = self.t / time_delta

        x = vehicle.x[timestep]
        y = vehicle.y[timestep]
        theta = vehicle.theta[timestep]
        v = vehicle.v[timestep]
        phi = vehicle.phi[timestep]

        x_new, y_new, theta_new, v_new, phi_new = run_equations(x, y, theta, v, phi, wheelbase, u1, u2, self.sess)

        self.t += time_delta
        timestep = self.t / time_delta

        vehicle.v[timestep] = v_new
        vehicle.phi[timestep] = phi_new
        self.update_xy_theta(x_new, y_new, theta_new, vehicle)

    def update_xy_theta(self, x, y, theta, vehicle):
        '''
        Updates the vehicle's x, y, and theta values if the vehicle will remain on the road and out of collision.
        A vehicle is on the road/lot if the intertial position of the rear axle lies on the road/lot.
        '''
        timestep = self.t / time_delta
        if not timestep or x < 0 or x >= self.x_bound or y < 0 or y > self.y_bound: return

        old_x = vehicle.x[timestep-1]
        old_y = vehicle.y[timestep-1]
        old_theta = vehicle.theta[timestep-1]

        vehicle.x[timestep] = x
        vehicle.y[timestep] = y
        vehicle.theta[timestep] = theta

        on_road = np.any(map(lambda r: r.is_on(x, y), self.roads))
        in_lot = np.any(map(lambda l: l.is_on(x, y), self.lots)) if not on_road else False

        if not (n_road or in_lot) or not check_all_collisions(self, vehicle):
            vehicle.x[timestep] = old_x
            vehicle.y[timestep] = old_y
            vehicle.theta[timestep] = old_theta
            return
        
        vehicle.update_xy_theta(x, y, theta, timestep)

    def check_all_collisions(self, vehicle):
        '''
        Checks if the vehicle is ever in collision at the current time
        '''
        timestep = self.t / time_delta

        return check_vehicle_collisions(vehicle, self.external_vehicles, timestep) or \
               check_vehicle_collisions(vehicle, self.obstacles, timestep) or \
               check_vehicle_collisions(vehicle, self.crates, timestep) or \
               check_vehicle_collisions(vehicle, self.user_vehicles, timestep)
