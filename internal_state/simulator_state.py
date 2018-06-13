import tensorflow as tf

from internal_state.collision_utils import *
from internal_state.constants import *
from internal_state.dynamics import *
from internal_state.objects.vehicle import Vehicle
from utils.poly_utils import *

class SimulatorState:
    '''
    Class to track the state of the driving simulation
    '''
    def __init__(self, horizon, x_bound, y_bound):
        self.horizon = horizon
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.external_vehicles = []
        self.crates = []
        self.obstacles = []
        self.stop_signs = []
        self.roads = []
        self.lots = []

        self.t = 0
        self.sess = tf.Session()

        self.user_vehicle = Vehicle(self, horizon, wheelbase=wheelbase, width=vehicle_width)

    def set_time(self, time):
        self.t = time

    def step(self, u1, u2, vehicle):
        '''
        Run the simulator forward one timestep for the given input controls for the given vehicle
        '''
        timestep = self.t / time_delta

        px = vehicle.px[timestep]
        py = vehicle.py[timestep]
        theta = vehicle.theta[timestep]
        v = vehicle.v[timestep]
        phi = vehicle.phi[timestep]

        px_new, py_new, theta_new, v_new, phi_new = run_equations(px, py, theta, v, phi, wheelbase, u1, u2, self.sess)

        self.t += time_delta
        timestep = self.t / time_delta

        vehicle.v[timestep] = v_new
        vehicle.phi[timestep] = phi_new
        self.update_xy_theta(px_new, py_new, theta_new, vehicle)

    def update_xy_theta(self, px, py, theta, vehicle):
        '''
        Updates the vehicle's x, y, and theta values if the vehicle will remain on the road and out of collision.
        A vehicle is on the road/lot if the intertial position of the rear axle lies on the road/lot.
        '''
        timestep = self.t / time_delta
        if not timestep: return

        old_px = vehicle.px[timestep-1]
        old_py = vehicle.py[timestep-1]
        old_theta = vehicle.theta[timestep-1]

        vehicle.px[timestep] = px
        vehicle.py[timestep] = py
        vehicle.theta[timestep] = theta

        on_road = np.any(map(lambda r: r.is_on_road(px, py), self.roads))
        in_lot = np.any(map(lambda l: l.is_in_lot(px, py), self.lots)) if not on_road else False

        if not (n_road or in_lot) or not check_all_collisions(self, vehicle):
            vehicle.px[timestep] = old_px
            vehicle.py[timestep] = old_py
            vehicle.theta[timestep] = old_theta
            return
        
        vehicle.update_xy_theta(px, py, theta, timestep)

    def check_all_collisions(self, vehicle):
        '''
        Checks if the vehicle is ever in collision at the current time
        '''
        timestep = self.t / time_delta

        return check_vehicle_collisions(vehicle, self.external_vehicles, timestep) or \
               check_vehicle_collisions(vehicle, self.obstacles, timestep) or \
               check_vehicle_collisions(vehicle, self.crates, timestep) or \
               check_vehicle_collisions(vehicle, [self.user_vehicle], timestep)
