from matplotlib.patches import Rectangle
import numpy as np

from internal_state.constants import *

class Road:
    '''
    Class representing a one-directional road
    '''

    def __init__(self, x, y, length, direction, num_lanes=num_lanes, lane_width=lane_width):
        '''
        x: The center of the road (halfway across lanes) along the east-west axis. Positive = east
        y: The center of the road (halfway across lanes) along the north-south axis. Positive = north
        direction: The angle of the road where east is 0 degrees and north is 90 degrees
        '''
        self.x = x
        self.y = y
        self.length = length
        self.direction = direction
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.rot_mat = np.array([[np.cos(direction), -np.sin(direction)], 
                                 [np.sin(direction), np.cos(direction)]])
        self.rot_origin = np.dot(self.rot_mat, np.array([self.x, self.y]))

    def get_lower_left(self):
        '''
        Return the coordinate of the road's lower left corner.
        Used for rendering purposes.
        '''
        vec = [-self.length / 2., -self.num_lanes * self.lane_width / 2.]
        return self.rot_mat.dot(vec) + [self.x, self.y]

    def get_lane_lower_lefts(self):
        '''
        Return the coordinate of each lane's lower left corner.
        Used for rendering purposes.
        '''
        coords = []
        for i in range(self.num_lanes):
            vec = [-self.length / 2., (i - self.num_lanes/2.) * self.lane_width]
            coords.append(self.rot_mat.dot(vec) + [self.x, self.y])

        return np.array(coords, dtype='int32')

    def get_patches(self):
        '''
        Return the Matplotlib patches for this road's lanes.
        Used for rendering purposes.
        '''
        lanes = self.get_lane_lower_lefts()
        for lane in lanes:
            rects.append(Rectanle(((lane[0], lane[1]), self.length, self.lane_width, self.direction)))
            
        return rects

    def is_on_road(self, x, y):
        '''
        Checks if a coordinate lies on the road.
        Points are put into the road rotational frame first, which treats the road as running west to east.
        These checks return true if the point is within half the width of the road and half the length of the road from
        the road origin, along the x- and y- axes respectively.
        '''
        rot_xy = np.dot(self.rot_mat, np.array([x, y]))
        return np.abs(rot_xy[1] - self.y) < 0.5 * self.num_lanes * self.lane_width and \
               np.abs(rot_xy[0] - self.x) < 0.5 * self.length
