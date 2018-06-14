import numpy as np

from internal_state.constants import *
from internal_state.surfaces.surface import DrivingSurface

class Lot:
    '''
    Class representing an open lot
    '''

    def __init__(self, x, y, height, width):
        '''
        x: The center of the lot along the east-west axis. Positive = east
        y: The center of the lot along the north-south axis. Positive = north
        height: The dimension of the lot along the north-south axis
        width: The dimension of the lot along the east-west axis
        '''
        self.width = width
        self.height = height
        
        super(Road, self).__init__(x, y)

    def is_on(self, x, y):
        '''
        Checks if a coordinate lies in the lot.
        '''
        return np.abs(x - self.x) <= self.width / 2.0 and np.abs(y - self.y) <= self.height / 2.0

    def to(self, x, y):
        '''
        Returns the shortest vector that shifts x, y onto the lot
        '''
        
        x_delta = 0
        y_delta = 0

        x_dist = x - self.x
        y_dist = y - self.y

        if x_dist > self.width: x_delta = -x_dist + self.width
        if x_dist < -self.width: x_delta = -x_dist - self.width
        if y_dist > self.height: y_delta = -y_dist + self.height
        if y_dist < -self.height: y_delta = -y_dist - self.height

        return np.array([x_delta, y_delta])
