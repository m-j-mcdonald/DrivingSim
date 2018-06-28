from matplotlib.patches import Rectangle
import numpy as np

from driving_utils.constants import *
from internal_state.surfaces.surface import DrivingSurface

class Road(DrivingSurface):
    '''
    Class representing a one-directional road
    '''

    def __init__(self, road_id, x, y, length, direction, num_lanes=num_lanes, lane_width=lane_width):
        '''
        x: The center of the road (halfway across lanes) along the east-west axis. Positive = east
        y: The center of the road (halfway across lanes) along the north-south axis. Positive = north
        direction: The angle of the road where east is 0 degrees and north is 90 degrees
        '''
        self.length = length
        self.direction = direction
        self.slope = np.tan(direction) if np.abs(direction - np.pi) < theta_tol and np.abs(direction - 3 * np.pi / 2) < theta_tol else np.inf
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.rot_mat = np.array([[np.cos(direction), -np.sin(direction)], 
                                 [np.sin(direction), np.cos(direction)]])
        self.inv_rot_mat = np.array([[np.cos(direction), np.sin(direction)], 
                                     [-np.sin(direction), np.cos(direction)]])
        self.rot_origin = np.dot(self.rot_mat, np.array([x, y]))

        super(Road, self).__init__(road_id, x, y)

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

    def get_patches(self, scale, unused):
        '''
        Return the Matplotlib patches for this road's lanes.
        Used for rendering purposes.
        '''
        lanes = self.get_lane_lower_lefts() * scale
        rects = []

        for lane in lanes:
            rects.append(Rectangle((lane[0], lane[1]), self.length * scale[0], self.lane_width * scale[1], self.direction, edgecolor='k', facecolor='w'))
            
        return rects

    def is_on(self, x, y):
        '''
        Checks if a coordinate lies on the road.
        Points are put into the road rotational frame first, which treats the road as running west to east.
        These checks return true if the point is within half the width of the road and half the length of the road from
        the road origin, along the x- and y- axes respectively.
        '''
        rot_xy = np.dot(self.rot_mat, np.array([x, y]))
        return np.abs(rot_xy[1] - self.y) < 0.5 * self.num_lanes * self.lane_width and \
               np.abs(rot_xy[0] - self.x) < 0.5 * self.length

    def to(self, x, y):
        '''
        Returns the shortest vector that shifts x, y onto the road
        '''
        rot_xy = np.dot(self.rot_mat, np.array([x, y]))
        if rot_xy[1] > 0.5 * self.num_lanes * self.lane_width + self.y:
            yvec_mult = -1
        elif rot_xy[1] < -0.5 * self.num_lanes * self.lane_width + self.y:
            yvec_mult = 1
        else:
            yvec_mult = 0
        
        if rot_xy[0] > 0.5 * self.length + self.x:
            xvec_mult = -1
        elif rot_xy[0] < -0.5 * self.length + self.x:
            xvec_mult = 1
        else:
            xvec_mult = 0

        vec = [rot_xy[0] + xvec_mult * 0.5 * self.length + self.x,
               rot_xy[1] + yvec_mult * 0.5 * self.num_lanes * self.lane_width + self.y]

        return self.inv_rot_mat.dot(vec)

    def to_lane(self, x, y, theta, lane_num):
        '''
        Returns the shortest vector that shifts x, y, theta to the center of the given lane facing down the road.
        Lanes are numbered starting from 0, with 0 being the leftmost lane.
        '''
        rot_xy = self.rot_mat.dot(np.array([x, y]))
        center_y = (self.num_lanes/2. - lane_num - 0.5) * self.lane_width + self.y
        center_x = self.x

        vec = [0, center_y - y]

        theta_diff = self.direction - theta

        while theta_diff > np.pi:
            theta_diff -= 2 * np.pi

        while theta_diff < -np.pi:
            theta_diff += 2 * np.pi

        return self.inv_rot_mat.dot(vec), theta_diff

    def get_intersection(self, road):
        if self.slope == road.slope:
            return (np.nan, np.nan)

        if np.isinf(self.slope):
            intersection = (self.x, road.slope * (self.x - road.x) + road.y)
        elif np.isinf(road.slope):
            intersection = (road.x, self.slope * (road.x - self.x) + self.y)
        else:
            x = (road.y - self.y - road.slope * road.x + self.slope * self.x) / (self.slope - road.slope)
            y = self.slope * (x - self.x) + self.y
            intersection = (x, y)

        if np.all(self.to(intersection == 0)) and np.all(road.to(intersection) == 0):
            return intersection

        return (np.nan, np.nan)

    def to_intersection(self, x, y, road):
        intersection = self.get_intersection(road)
        if np.any(np.isnan(intersection)):
            return np.inf, np.inf

        return intersection[0] - x, intersection[1] - y
        
    def at_intersection(self, x, y, road, dist=stopping_len):
        road_width = road.num_lanes * road.lane_width / 2.
        to_intersection = np.linalg.norm(self.to_intersection(x, y, road))
        is_on = self.is_on(x, y)

        in_intersection = is_on and to_intersection < road_width
        stopped_at_intersection = not in_intersection and is_on and to_intersection < road_width + dist
        return in_intersection, stopped_at_intersection
        