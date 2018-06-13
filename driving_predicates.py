import numpy as np

from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

from core.util_classes.common_predicates import ExprPredicate

from internal_state.dynamics import *

class DynamicPredicate(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int)),
                                             ("vel", np.array([0], dtype=np.int)),
                                             ("phi", np.array([0], dtype=np.int)),
                                             ("u1", np.array([0], dtype=np.int)),
                                             ("u2", np.array([0], dtype=np.int))])])

        val = np.zeros((1, 14))
        dynamics_expr = Expr(self.f, self.grad)
        e = EqExpr(dynamics_expr, val)

        super(DynamicPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class XValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return np.array([x[7] - next_px_f(x[0], x[2], x[3], x[5], x[6], self.sess)])

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,7] = 1
            grad[0,5], grad[0,6] = 
            next_px_grad(x[0], x[2], x[3], x[5], x[6], self.sess)
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

class Stationary(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class VehicleStationary(Stationary):
    pass

class CrateStationary(Stationary):
    pass

class ObstacleStationary(Stationary):
    pass

