import numpy as np

from driving_sim.internal_state.constants import *
from driving_sim.internal_state.surfaces.surface import DrivingSurface

class Lot(DrivingSurface):
    '''
    Class representing an open lot
    '''

    def __init__(self, lot_id, x, y, height, width):
        '''
        x: The center of the lot along the east-west axis. Positive = east
        y: The center of the lot along the north-south axis. Positive = north
        height: The dimension of the lot along the north-south axis
        width: The dimension of the lot along the east-west axis
        '''
        self.width = width
        self.height = height
        
        super(Lot, self).__init__(lot_id, x, y)

    def get_patches(self, scale, unused):
        '''
        Return the Matplotlib patches for this lot.
        Used for rendering purposes.
        '''
        for lane in lanes:
            rects.append(Rectanle(((lane[0], lane[1]), self.length, self.lane_width, self.direction), edgecolor='k'))
            
        return [Rectanle(((self.x - int(self.width / 2)) * scale[0], \
                          (self.y - int(self.height / 2)) * scale[1], \
                          self.width * scale[0], \
                          self.height * scale[1], \
                          0 \
                ))]

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
