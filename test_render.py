from internal_state.surfaces.road import Road
from internal_state.simulator_state import *
from driving_gui.gui import *

sim = SimulatorState(30, 100, 80)
sim.add_road(Road(50, 50, 100, 0))

g = GUI(sim)
g.draw_timestep(0)

