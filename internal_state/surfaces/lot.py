import numpy as np

from internal_state.constants import *

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
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_in_lot(self, x, y):
        '''
        Checks if a coordinate lies in the lot.
        '''
        return np.abs(x - self.x) <= self.width / 2.0 and np.abs(y - self.y) <= self.height / 2.0
