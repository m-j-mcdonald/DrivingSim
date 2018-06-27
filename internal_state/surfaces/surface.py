import numpy as np

class DrivingSurface(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_patches(self, unused):
        '''
        Return the Matplotlib patches for this surface at a given timestep.
        Used for rendering purposes.
        '''
        raise NotImplementedError

    def to(self, x, y):
        '''
        Returns the shortest vector that shifts x, y onto the surface
        '''
        raise NotImplementedError

    def is_on(self, x, y):
        '''
        Checks if a coordinate lies on the surface.
        '''
        raise NotImplementedError
