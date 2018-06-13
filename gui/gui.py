from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from gui.patch_utils import *
from internal_state.constants import *

class GUI:
    '''
    Controls the simulator graphics
    '''
    def __init__(self, sim_state, x_bound, y_bound):
        self.state = sim_state
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.fig = plt.figure()
        self.ax = fig.add_subplot(x_bound, y_bound, 1)

    def run_sim(self, start_t=0, real_t=0.1):
        '''
        Runs the simulator from the starting timestep until the end
        start_t: The time to start the simulator from
        real_t: How long the simulator pauses between timesteps
        '''
        start_timestep = start_t / time_delta
        end_timestep = self.state.horizon / time_delta
        interval = int(1000 * real_t)
        frames = rabge(start_timestep, end_timestep+1)
        animation = FuncAnimation(self.fig, self.draw_timestep, frames=frames, interval=interval)

    def draw_timestep(self, time):
        '''
        Renders the simulator's state at a given timestep
        '''
        self.clear()
        self.add_objects(time)
        self.add_surfaces()
        plt.show()

    def add_collection(self, to_render, time=-1):
        '''
        Add all patches for a given set of objects or surfaces
        '''
        patches = []
        for r in to_render:
            patches.extend(r.get_patches(time)) if time >= 0 else patches.extend(r.get_patches())

        col = PatchCollection(patches)
        self.ax.add_collection(col)

    def add_vehicles(self, time):
        '''
        Add all vehicles to be rendered at a given timestep
        '''
        vehicles = self.sim_state.external_vehicles + self.sim_state.user_vehicle
        self.add_collection(vehicles, time)

    def add_crates(self, time):
        '''
        Add all crates to be rendered at a given timestep
        '''
        crates = self.sim_state.crates
        self.add_collection(crates, time)

    def add_obstacles(self, time):
        '''
        Add all obstacles to be rendered at a given timestep
        '''
        obstacles = self.sim_state.obstacles
        self.add_collection(obstacles, time)

    def add_roads(self):
        '''
        Add all roads to be rendered at a given timestep
        '''
        roads = self.sim_state.roads
        self.add_collection(roads)

    def add_lots(self):
        '''
        Add all lots to be rendered at a given timestep
        '''
        lots = self.sim_state.lots
        self.add_collection(lots)

    def add_objects(self, time):
        '''
        Add all objects to be rendered at a given timestep
        '''
        self.add_vehicles(time)
        self.add_crates(time)
        self.add_obstacles(time)

    def add_surfaces(self):
        '''
        Add all surfaces to be rendered
        '''
        self.add_roads()
        self.add_lots()

    def clear(self):
        '''
        Clears the previously rendered drawing
        '''
        self.ax.clear()
