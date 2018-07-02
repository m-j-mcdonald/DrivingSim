from internal_state.objects.vehicle import Vehicle
from internal_state.surfaces.road import Road
from internal_state.simulator_state import *
from driving_gui.gui import *

sim = SimulatorState(30, 100, 80)
sim.add_road(Road(0, 50, 40, 100, 0))

v = Vehicle(10, 5, 40, 0, is_user=True)
sim.add_vehicle(v)

v.update_xy_theta(6, 40, 0, 1)
v.update_xy_theta(7, 40, 0.1, 2)
v.update_xy_theta(8, 40, 0.2, 3)
v.update_xy_theta(9, 40, 0.3, 4)
v.update_xy_theta(10, 40, 0.4, 5)
v.update_xy_theta(11, 40, 0.5, 6)
v.update_xy_theta(12, 40, 0.6, 7)
v.update_xy_theta(13, 40, 0.7, 8)
v.update_xy_theta(14, 40, 0.8, 8)

g = GUI(sim)
# g.draw_timestep(0)
#plt.show()
g.run_sim()
