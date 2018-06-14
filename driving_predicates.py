import numpy as np

from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

from core.util_classes.common_predicates import ExprPredicate

from internal_state.dynamics import *

MOVE_FACTOR = 0.5

class DynamicPredicate(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int)),
                                             ("vel", np.array([0], dtype=np.int)),
                                             ("phi", np.array([0], dtype=np.int)),
                                             ("u1", np.array([0], dtype=np.int)),
                                             ("u2", np.array([0], dtype=np.int))])])

        val = np.zeros((1, 14))
        dynamics_expr = Expr(self.f, self.grad)
        e = EqExpr(dynamics_expr, val)

        super(DynamicPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False

class XValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[7] - next_px_f(x[0], x[2], x[3], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,7] = 1
            grad[0,5], grad[0,6] = next_px_grad(x[0], x[2], x[3], x[5], x[6], self.sess)
            return grad

        self.f = f
        self.grad = grad
        super(XValid).__init__(name, params, expected_param_types, env)

class YValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[8] - next_py_f(x[1], x[2], x[3], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,8] = 1
            grad[0,5], grad[0,6] = next_py_grad(x[1], x[2], x[3], x[5], x[6], self.sess)
            return grad

        self.f = f
        self.grad = grad
        super(YValid).__init__(name, params, expected_param_types, env)

class ThetaValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[9] - next_theta_f(x[2], x[3], x[4], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,9] = 1
            grad[0,5], grad[0,6] = next_theta_grad(x[2], x[3], x[4], x[5], x[6], self.sess)
            return grad

        self.f = f
        self.grad = grad
        super(ThetaValid).__init__(name, params, expected_param_types, env)

class VelocityValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[10] - next_v_f(x[3], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,10] = 1
            grad[0,5], grad[0,6] = next_v_grad(x[3], x[5], x[6], self.sess)
            return grad

        self.f = f
        self.grad = grad
        super(VelocityValid).__init__(name, params, expected_param_types, env)
        
class PhiValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[11] - next_phi_f(x[4], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,11] = 1
            grad[0,5], grad[0,6] = next_phi_grad(x[4], x[5], x[6], self.sess)
            return grad

        self.f = f
        self.grad = grad
        super(PhiValid).__init__(name, params, expected_param_types, env)
        
class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))]),
                                 (self.target, [("xy", np.array([0,1], dtype=np.int)),
                                                ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
    
class VehicleAt(At):
    pass

class CrateAt(At):
    pass

class ObstacleAt(At):
    pass

class Near(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.target, self.dist = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))]),
                                 (self.target, [("xy", np.array([0,1], dtype=np.int))]),,
                                 (self.dist, [("value", np.array([0], dtype=np.int))])])

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)], -np.ones((4,1))]
        b, val = np.zeros((4, 1)), np.zeros((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True

class VehicleAtSign(Near):
    pass

class Stationary(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False

class VehicleStationary(Stationary):
    pass

class CrateStationary(Stationary):
    pass

class ObstacleStationary(Stationary):
    pass

class IsMP(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b, val = np.zeros((6, 1)), MOVE_FACTOR * np.ones((6, 1))
        e = LEqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False

class OnSurface(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.surface = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])

        f = lambda x: self.surface.body.to(x[0], x[1])
        grad = lambda x: np.eye(2)

        val = np.zeros((1, 2))
        dynamics_expr = Expr(self.f, self.grad)
        e = EqExpr(dynamics_expr, val)

        super(OnSurface, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = False

class OnRoad(OnSurface):
    pass

class OnLot(ExprPredicate):
    pass

class InLane(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.road, self.lane_num = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))]),
                                 (self.lane_num, [("value", np.array([0], dtype=np.int))])])

        f = lambda x: self.road.body.to_lane(x[0], x[1], x[2])
        grad = lambda x: np.eye(3)

        val = np.zeros((1, 2))
        dynamics_expr = Expr(self.f, self.grad)
        e = EqExpr(dynamics_expr, val)

        super(OnSurface, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = False

class Limit(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.limit = params
        attr_inds = OrderedDict([(self.obj, [("v", np.array([0], dtype=np.int)),
                                             ("u1", np.array([0], dtype=np.int))]),
                                 (self.limit, [("value", np.array([0], dtype=np.int))])])

        b, val = np.zeros((1, 1)), np.zeros((1, 1))
        e = LEqExpr(AffExpr(self.A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = False

class VelLowerLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,0] = -1
        self.A[0,2] = 1
        super(VelLowerLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class VelUpperLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,0] = 1
        self.A[0,2] = -1
        super(VelUpperLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class AccLowerLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,1] = -1
        self.A[0,2] = 1
        super(AccLowerLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class AccUpperLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,1] = 1
        self.A[0,2] = -1
        super(AccUpperLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj1, self.obj2 = params
        attr_inds = OrderedDict([(self.obj1, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))]),
                                 (self.obj2, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))])])

        def f(x):
            old_pose1 = self.obj1.body.update_xy_theta(x[0], x[1], x[2], 0)
            old_pose2 = self.obj2.body.update_xy_theta(x[3], x[4], x[5], 0)
            obj1_pts = self.obj1.body.get_points()
            obj2_pts = self.obj2.body.get_points()
            self.obj1.body.update_xy_theta(0, old_pose1[0], old_pose1[1], old_pose1[2])
            self.obj2.body.update_xy_theta(0, old_pose2[0], old_pose2[1], old_pose2[2])
            return collision_vector(obj1_pts, obj2_pts)
        
        def grad(obj1_body, obj2_body):
            grad = np.zeros((2,6))
            grad[:, :2] = -np.eye(2)
            grad[:, 3:5] = np.eye(2)

        val = np.zeros((1, 2))
        col_expr = Expr(f, grad)
        e = EqExpr(col_expr, val)

        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = False

class VehicleVehicleCollision(CollisionPredicate):
    pass

class VehicleObstacleCollision(CollisionPredicate):
    pass

class VehicleCrateCollision(CollisionPredicate):
    pass
