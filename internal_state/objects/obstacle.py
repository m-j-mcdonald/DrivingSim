import numpy as np
from scipy.spatial import ConvexHull

class Obstacle:
    '''
    Class representing an obstacle vehicles must avoid
    '''
    def __init__(self, x, y, points, obstacle_id):
        '''
        x: The x coordinate which the obstacle starts at and which all points are reference to
        y: The y coordinate which the obstacle starts at and which all points are reference to
        points: The defining points of the obstacle as vectors from x, y
        obstacle_id: An unique identifier for the obstacle
        '''
        self.x = x * np.ones(horizon, dtype='float32')
        self.y = y * np.ones(horizon, dtype='float32')

        #Only keep the points that define the convex hull
        hull = ConvexHull(points)
        self.points = hull.points[hull.vertices]

        self.id = obstacle_ids

    def get_points(self, time):
        '''
        Returns the coordinates of the obstacle's defining points at a given timestep
        '''
        return np.array(self.points) + np.array([self.x[time], self.y[time]])

    def get_patches(self, time):
        '''
        Return the Matplotlib patches for this obstacle at a given timestep.
        Used for rendering purposes.
        '''
        return [Polygon(self.get_points(time))]

    def get_hull(self, time):
        '''
        Returns the convex hull of the points defining the obstacle at a given timestep
        '''
        points = self.get_points(time)
        return ConvexHull(points)
