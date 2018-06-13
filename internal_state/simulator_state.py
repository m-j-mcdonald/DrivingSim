import tensorflow as tf

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

        self.user_vehicle = Vehicle(wheelbase, vehicle_width)
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
        self.update_xy_if_on_road(px_new, py_new, theta_new, vehicle)

    def update_xy_theta_if_on_road(self, px, py, theta, vehicle):
        '''
        A vehicle is on the road/lot if the intertial position of the rear axle lies on the road/lot
        '''
        timestep = self.t / time_delta
        if np.any(map(lambda r: r.is_on_road(px, py), self.roads)) or np.any(map(lambda l: l.is_in_lot(px, py), self.lots)):
            vehicle.update_xy_theta(px, py, theta, timestep)

    def check_collision(self, vehicle):
        '''
        Take Minkowski difference between vehicle and all possible obstructions in order to determine if in collision
        '''
        if vehicle is not self.user_vehicle:
            pass

        for v in self.external_vehicles:
            if vehicle is not v:
                pass
