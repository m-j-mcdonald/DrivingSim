import sys
sys.path.insert(0, '../../src/')
dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Vehicle, Road, Obstacle, Crate, VehiclePose, CratePose, ObstaclePose, StopSign, Distance, Limit, LaneNumber

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: Vehicle internal_state.objects.vehicle, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Vector3d core.util_classes.matrix, Obstacle internal_state.objects.obstacle, Crate internal_state.objects.crate, Road internal_state.surfaces.road, StopSign internal_state.signs.stop_sign

Predicates Import Path: driving_predicates

"""

class PrimitivePredicates(object):
    def __init__(self):
        self.attr_dict = {}

    def add(self, name, attrs):
        self.attr_dict[name] = attrs

    def get_str(self):
        prim_str = 'Primitive Predicates: '
        first = True
        for name, attrs in self.attr_dict.iteritems():
            for attr_name, attr_type in attrs:
                pred_str = attr_name + ', ' + name + ', ' + attr_type
                if first:
                    prim_str += pred_str
                    first = False
                else:
                    prim_str += '; ' + pred_str
        return prim_str

pp = PrimitivePredicates()
pp.add('Vehicle', [('geom', 'Vehicle'), ('xy', 'Vector2d'), ('theta', 'Vector1d'), ('vel', 'Vector1d'), ('phi', 'Vector1d'), ('u1', 'Vector1d'), ('u2', 'Vector1d')])
pp.add('Obstacle', [('geom', 'Obstacle'), ('xy', 'Vector2d'), ('theta', 'Vector1d')])
pp.add('Crate', [('geom', 'Crate'), ('xy', 'Vector2d'), ('theta', 'Vector1d')])
pp.add('Road', [('geom', 'Road'), ('xy', 'Vector2d'), ('theta', 'Vector1d')])
pp.add('StopSign', [('geom', 'StopSign'), ('xy', 'Vector2d')])
pp.add('VehiclePose', [('value', 'Vector1d'), ('xy', 'Vector2d'), ('theta', 'Vector1d'), ('vel', 'Vector1d'), ('phi', 'Vector1d'), ('u1', 'Vector1d'), ('u2', 'Vector1d')])
pp.add('ObstaclePose', [('value', 'Vector1d'), ('xy', 'Vector2d'), ('theta', 'Vector1d')])
pp.add('CratePose', [('value', 'Vector1d'), ('xy', 'Vector2d'), ('theta', 'Vector1d')])
pp.add('Distance', [('value', 'Vector1d')])
pp.add('Limit', [('value', 'Vector1d')])
pp.add('LaneNumber', [('value', 'Vector1d')])

dom_str += pp.get_str() + '\n\n'

class DerivatedPredicates(object):
    def __init__(self):
        self.pred_dict = {}

    def add(self, name, args):
        self.pred_dict[name] = args

    def get_str(self):
        prim_str = 'Derived Predicates: '

        first = True
        for name, args in self.pred_dict.iteritems():
            pred_str = name
            for arg in args:
                pred_str += ', ' + arg

            if first:
                prim_str += pred_str
                first = False
            else:
                prim_str += '; ' + pred_str
        return prim_str

dp = DerivatedPredicates()
dp.add('HLNoCollisions', ['Vehicle'])
dp.add('HLCrateInTrunk', ['Vehicle', 'Crate'])

dp.add('VehicleAt', ['Vehicle', 'VehiclePose'])
dp.add('ObstacleAt', ['Obstacle', 'ObstaclePose'])
dp.add('CrateAt', ['Crate', 'CratePose'])
dp.add('VehicleAtSign', ['Vehicle', 'StopSign'])
dp.add('VehicleVelAt', ['Vehicle', 'Limit'])
dp.add('ExternalVehicleVelAt', ['Vehicle', 'Limit'])
dp.add('XValid', ['Vehicle'])
dp.add('YValid', ['Vehicle'])
dp.add('ThetaValid', ['Vehicle'])
dp.add('VelValid', ['Vehicle'])
dp.add('PhiValid', ['Vehicle'])
dp.add('VehicleStationary', ['Vehicle'])
dp.add('ObstacleStationary', ['Obstacle'])
dp.add('CrateStationary', ['Crate'])
dp.add('IsMP', ['Vehicle'])
dp.add('OnRoad', ['Vehicle', 'Road'])
dp.add('InLane', ['Vehicle', 'Road', 'LaneNumber'])
dp.add('ExternalInLane', ['Vehicle', 'Road', 'LaneNumber'])
dp.add('LeftOfLane', ['Vehicle', 'Road', 'LaneNumber'])
dp.add('RightOfLane', ['Vehicle', 'Road', 'LaneNumber'])
dp.add('PoseInLane', ['VehiclePose', 'Road', 'LaneNumber'])
dp.add('PoseLeftOfLane', ['VehiclePose', 'Road', 'LaneNumber'])
dp.add('PoseRightOfLane', ['VehiclePose', 'Road', 'LaneNumber'])
dp.add('VelLowerLimit', ['Vehicle', 'Limit'])
dp.add('VelUpperLimit', ['Vehicle', 'Limit'])
dp.add('AccLowerLimit', ['Vehicle', 'Limit'])
dp.add('AccUpperLimit', ['Vehicle', 'Limit'])
dp.add('VehicleVehicleCollision', ['Vehicle', 'Vehicle'])
dp.add('VehicleObstacleCollision', ['Vehicle', 'Obstacle'])
dp.add('VehicleCrateCollision', ['Vehicle', 'Crate'])
dp.add('CrateObstacleCollision', ['Crate', 'Obstacle'])
dp.add('VehicleVehiclePathCollision', ['Vehicle', 'Vehicle'])
dp.add('VehicleObstaclePathCollision', ['Vehicle', 'Obstacle'])
dp.add('VehicleCratePathCollision', ['Vehicle', 'Crate'])
dp.add('CrateObstaclePathCollision', ['Crate', 'Obstacle'])
dp.add('Follow', ['Vehicle', 'Vehicle', 'Distance'])
dp.add('StopAtStopSign', ['Vehicle', 'StopSign'])
dp.add('ExternalVehiclePastRoadEnd', ['Vehicle', 'Road'])
dp.add('ExternalDriveDownRoad', ['Vehicle', 'Road'])
dp.add('PosesWithinDistance', ['VehiclePose', 'VehiclePose', 'Distance'])

dom_str += dp.get_str() + '\n'

dom_str += """

# The first set of parentheses after the colon contains the
# parameters. The second contains preconditions and the third contains
# effects. This split between preconditions and effects is only used
# for task planning purposes. Our system treats all predicates
# similarly, using the numbers at the end, which specify active
# timesteps during which each predicate must hold

"""

class Action(object):
    def __init__(self, name, timesteps, pre=None, post=None):
        pass

    def to_str(self):
        time_str = ''
        cond_str = '(and '
        for pre, timesteps in self.pre:
            cond_str += pre + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        cond_str += '(and '
        for eff, timesteps in self.eff:
            cond_str += eff + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        return "Action " + self.name + ' ' + str(self.timesteps) + ': ' + self.args + ' ' + cond_str + ' ' + time_str

class DrivingAction(Action):
    def __init__(self, name, timesteps, pre, post, args):
        self.name = name
        self.timesteps = timesteps
        end = timesteps - 1

        self.args = args

        self.pre = [\
            ('(VehicleAt ?vehicle ?start)', '{}:{}'.format(0, 0)),
            ('(forall (?obj - Vehicle)\
                (XValid ?vehicle))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (YValid ?vehicle))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (ThetaValid ?vehicle))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (VelValid ?vehicle))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (PhiValid ?vehicle))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (VehicleVehiclePathCollision ?vehicle ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Obstacle)\
                (VehicleObstaclePathCollision ?vehicle ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Crate)\
                (VehicleCratePathCollision ?vehicle ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Vehicle)\
                (ExternalDriveDownRoad ?vehicle))', '{}:{}'.format(0, end-1)),
        ]
        self.pre.extend(pre)

        self.eff = [\
            (' (not (VehicleAt ?vehicle ?start))', '{}:{}'.format(end, end-1)),
            ('(VehicleAt ?vehicle ?end)', '{}:{}'.format(end, end)),
            ('(forall (?obj - Vehicle)\
                (ExternalVehiclePastRoadEnd ?vehicle))', '{}:{}'.format(end, end)),
        ]
        self.eff.extend(post)

class DriveDownRoad(DrivingAction):
    def __init__(self):
        name = 'drive_down_road'
        timesteps = 100
        end = timesteps - 1
        args = '(?vehicle - Vehicle ?road - Road ?start - VehiclePose ?end - VehiclePose ?vu_limit - Limit ?vl_limit - Limit ?au_limit - Limit ?al_limit - Limit)'
        
        pre = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelUpperLimit ?vehicle ?vu_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelLowerLimit ?vehicle ?vl_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccUpperLimit ?vehicle ?au_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccLowerLimit ?vehicle ?al_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Obstacle) (ObstacleStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?obj - Crate) (CrateStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?sign - StopSign)\
                    (forall (?obj - Vehicle)\
                        (StopAtStopSign ?obj ?sign)))', '{}:{}'.format(0, end-1)),
              ]

        eff = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(end, end-1)),
              ]

        super(DriveDownRoad, self).__init__(name, timesteps, pre, eff, args)

class Follow(Action):
    def __init__(self):
        name = 'follow'
        timesteps = 100
        end = timesteps - 1
        args = '(?vehicle - Vehicle ?external - Vehicle ?road - Road \
                 ?start - VehiclePose ?end - VehiclePose ?external_start - VehiclePose \
                 ?lane_n - LaneNumber ?vu_limit - Limit ?vl_limit - Limit \
                 ?au_limit - Limit ?al_limit - Limit)'      

        pre = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(0, end)),
                ('(InLane ?vehicle ?road ?lane_n)', '{}:{}'.format(0, end)),
                ('(InLane ?external ?road ?lane_n)', '{}:{}'.format(0, end)),
                ('(PoseInLane ?end ?road ?lane_n)', '{}:{}'.format(0, 0)),
                ('(forall (?obj - Vehicle)\
                    (VelUpperLimit ?vehicle ?vu_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelLowerLimit ?vehicle ?vl_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccUpperLimit ?vehicle ?au_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccLowerLimit ?vehicle ?al_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Obstacle) (ObstacleStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?obj - Crate) (CrateStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?sign - StopSign)\
                    (forall (?obj - Vehicle)\
                        (StopAtStopSign ?obj ?sign)))', '{}:{}'.format(0, end-1)),
              ]

        eff = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(end, end-1)),
                ('(InLane ?vehicle ?road ?lane_n)', '{}:{}'.format(end, end-1)),
              ]

        super(Follow, self).__init__(name, timesteps, pre, eff, args)

class Pass(Action):
    def __init__(self):
        name = 'pass'
        timesteps = 100
        end = timesteps - 1
        args = '(?vehicle - Vehicle ?external - Vehicle ?road - Road \
                 ?start - VehiclePose ?end - VehiclePose ?external_start - VehiclePose \
                 ?lane_n - LaneNumber ?vu_limit - Limit ?vl_limit - Limit \
                 ?au_limit - Limit ?al_limit - Limit)'      

        pre = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(0, end)),
                ('(InLane ?external ?road ?lane_n)', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelUpperLimit ?vehicle ?vu_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelLowerLimit ?vehicle ?vl_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccUpperLimit ?vehicle ?au_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccLowerLimit ?vehicle ?al_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Obstacle) (ObstacleStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?obj - Crate) (CrateStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?sign - StopSign)\
                    (forall (?obj - Vehicle)\
                        (StopAtStopSign ?obj ?sign)))', '{}:{}'.format(0, end-1)),
              ]

        eff = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(end, end-1)),
              ]

        super(Pass, self).__init__(name, timesteps, pre, eff, args)

class Left(Action):
    def __init__(self):
        name = 'left'
        timesteps = 50
        end = timesteps - 1
        args = '(?vehicle - Vehicle ?road - Road ?start - VehiclePose ?end - VehiclePose \
                 ?lane_n - LaneNumber ?merge_dist - Distance \
                 ?vu_limit - Limit ?vl_limit - Limit ?au_limit - Limit ?al_limit - Limit)'      

        pre = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(0, end)),
                ('(PoseLeftOfLane ?end ?road ?lane_n)', '{}:{}'.format(end, end)),
                ('(PosesWithinDistance ?start ?end ?merge_dist)', '{}:{}'.format(0, 0))
                ('(forall (?obj - Vehicle)\
                    (VelUpperLimit ?vehicle ?vu_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelLowerLimit ?vehicle ?vl_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccUpperLimit ?vehicle ?au_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccLowerLimit ?vehicle ?al_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Obstacle) (ObstacleStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?obj - Crate) (CrateStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?sign - StopSign)\
                    (forall (?obj - Vehicle)\
                        (StopAtStopSign ?obj ?sign)))', '{}:{}'.format(0, end-1)),
              ]

        eff = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(end, end-1)),
                ('(LeftOfLane ?vehicle ?road ?lane_n)', '{}:{}'.format(end, end)),
              ]

        super(Left, self).__init__(name, timesteps, pre, eff, args)

class Right(Action):
    def __init__(self):
        name = 'right'
        timesteps = 50
        end = timesteps - 1
        args = '(?vehicle - Vehicle ?road - Road ?start - VehiclePose ?end - VehiclePose \
                 ?lane_n - LaneNumber ?merge_dist - Distance \
                 ?vu_limit - Limit ?vl_limit - Limit ?au_limit - Limit ?al_limit - Limit)'      

        pre = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(0, end)),
                ('(PoseRightOfLane ?end ?road ?lane_n)', '{}:{}'.format(end, end)),
                ('(PosesWithinDistance ?start ?end ?merge_dist)', '{}:{}'.format(0, 0))
                ('(forall (?obj - Vehicle)\
                    (VelUpperLimit ?vehicle ?vu_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (VelLowerLimit ?vehicle ?vl_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccUpperLimit ?vehicle ?au_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Vehicle)\
                    (AccLowerLimit ?vehicle ?al_limit))', '{}:{}'.format(0, end)),
                ('(forall (?obj - Obstacle) (ObstacleStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?obj - Crate) (CrateStationary ?obj)))', '{}:{}'.format(0, end-1)),
                ('(forall (?sign - StopSign)\
                    (forall (?obj - Vehicle)\
                        (StopAtStopSign ?obj ?sign)))', '{}:{}'.format(0, end-1)),
              ]

        eff = [\
                ('(OnRoad ?vehicle ?road)', '{}:{}'.format(end, end-1)),
                ('(RightOfLane ?vehicle ?road ?lane_n)', '{}:{}'.format(end, end)),
              ]

        super(Right, self).__init__(name, timesteps, pre, eff, args)

# TODO: StopAtSign, Turn, Wait
# tODO: Variants with grasping, shipping
