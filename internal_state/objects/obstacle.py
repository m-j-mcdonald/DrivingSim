import numpy as np
from scipy.spatial import ConvexHull

from internal_state.objects.object import DrivingObject

class Obstacle(DrivingObject):
    '''
    Class representing an obstacle vehicles must avoid
    '''
    def __init__(self, obj_id, x, y, horizon, points, obstacle_id):
        '''
        x: The x coordinate which the obstacle starts at and which all points are reference to
        y: The y coordinate which the obstacle starts at and which all points are reference to
        points: The defining points of the obstacle as vectors from x, y
        obstacle_id: An unique identifier for the obstacle
        '''

        #Only keep the points that define the convex hull
        hull = ConvexHull(points)
        self.points = hull.points[hull.vertices]
        self.color  = 'k'
        self.id     = obstacle_ids

        super(Obstacle, self).__init__(obj_id, x, y, 0, horizon)

    def get_points(self, time, dist=0):
        '''
        Returns the coordinates of the obstacle's defining points at a given timestep
        '''
        return np.array(self.points) + np.array([self.x[time], self.y[time]])
