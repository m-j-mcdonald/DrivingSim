'''
Definitions to perform dynamics calculations
'''
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from driving_sim.driving_utils.constants import time_delta

sess = tf.Session()

tf_px = tf.Variable(0, 'px', dtype='float32')
tf_py = tf.Variable(0, 'py', dtype='float32')
tf_theta = tf.Variable(0, 'theta', dtype='float32')
tf_v = tf.Variable(0, 'theta', dtype='float32')
tf_phi = tf.Variable(0, 'theta', dtype='float32')
tf_wheelbase = tf.Variable(0, 'wheelbase', dtype='float32')
tf_time_delta = tf.Variable(time_delta, dtype='float32', name='time_delta')

tf_u1 = tf.Variable(0, 'u1', dtype='float32') # v_dot
tf_u2 = tf.Variable(0, 'u2', dtype='float32') # phi_dot
tf_v_new = tf.add(tf_v, tf.multiply(tf_time_delta, tf_u1))
tf_phi_new = tf.add(tf_phi, tf.multiply(tf_time_delta, tf_u2))

tf_theta_dot = tf.multiply(tf_v_new, tf.div(tf.tan(tf_phi_new), tf_wheelbase))
tf_theta_new = tf.add(tf_theta, tf.multiply(tf_time_delta, tf_theta_dot))

tf_px_dot = tf.multiply(tf_v_new, tf.cos(tf_theta_new))
tf_py_dot = tf.multiply(tf_v_new, tf.sin(tf_theta_new))
tf_px_new = tf.add(tf_px, tf.multiply(tf_time_delta, tf_px_dot))
tf_py_new = tf.add(tf_py, tf.multiply(tf_time_delta, tf_py_dot))

def parse_grad(grad):
    for i in range(len(grad)):
      if grad[i] is None: grad[i] = tf.constant(0)

    return grad

grad_name_to_var = OrderedDict([('px', tf_px), ('py', tf_py), ('theta', tf_theta), ('vel', tf_v), ('phi', tf_phi), \
                   ('u1', tf_u1), ('u2', tf_u2), \
                   ('px_new', tf_px_new), ('py_new', tf_py_new), ('theta_new', tf_theta_new), \
                   ('v_new', tf_v_new), ('phi_new', tf_phi_new), ('px_dot', tf_px_dot), ('py_dot', tf_py_dot), \
                   ('theta_dot', tf_theta_dot)])

grad_var_to_name = {v: k for k, v in grad_name_to_var.iteritems()}

grad_variables = grad_name_to_var.values()
grad_names = grad_name_to_var.keys()

tf_px_new_grad = parse_grad(tf.gradients(tf_px_new, grad_variables))
tf_py_new_grad = parse_grad(tf.gradients(tf_py_new, grad_variables))
tf_theta_new_grad = parse_grad(tf.gradients(tf_theta_new, grad_variables))
tf_v_new_grad = parse_grad(tf.gradients(tf_v_new, grad_variables))
tf_phi_new_grad = parse_grad(tf.gradients(tf_phi_new, grad_variables))

init_op = tf.global_variables_initializer()
sess.run(init_op)

def select_grads(grad, tf_vars):
    selected_grad = []
    for v in tf_vars:
        for i in range(len(grad_variables)):
            if grad_variables[i] is v: selected_grad.append(grad[i])
    return selected_grad

def run_equations(px, py, theta, v, phi, wheelbase, u1, u2):
    return sess.run([tf_px_new, tf_py_new, tf_theta, tf_v_new, tf_phi_new], feed_dict={tf_px: px,
                                                                                       tf_py: py,
                                                                                       tf_theta: theta,
                                                                                       tf_v: v,
                                                                                       tf_phi: phi,
                                                                                       tf_wheelbase: wheelbase,
                                                                                       tf_u1: u1,
                                                                                       tf_u2: u2})

def next_px_f(wheelbase, px, theta, v, u1, u2):
    return sess.run(tf_px_new, feed_dict={tf_wheelbase: wheelbase, tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_px_grad(wheelbase, px, theta, v, u1, u2):
    grad = select_grads(tf_px_new_grad, [tf_px, tf_theta, tf_v, tf_u1, tf_u2])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_f(wheelbase, py, theta, v, u1, u2):
    return sess.run(tf_py_new, feed_dict={tf_wheelbase: wheelbase, tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_grad(wheelbase, py, theta, v, u1, u2):
    grad = select_grads(tf_py_new_grad, [tf_py, tf_theta, tf_v, tf_u1, tf_u2])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_theta_f(wheelbase, theta, v, phi, u1, u2):
    return sess.run(tf_theta_new, feed_dict={tf_wheelbase: wheelbase, tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_theta_grad(wheelbase, theta, v, phi, u1, u2):
    grad = select_grads(tf_theta_new_grad, [tf_theta, tf_v, tf_phi, tf_u1, tf_u2])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_v_f(wheelbase, v, u1, u2,):
    return sess.run(tf_v_new, feed_dict={tf_wheelbase: wheelbase, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_v_grad(wheelbase, v, u1, u2,):
    grad = select_grads(tf_v_new_grad, [tf_v, tf_u1, tf_u2])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_phi_f(wheelbase, phi, u1, u2):
    return sess.run(tf_phi_new, feed_dict={tf_wheelbase: wheelbase, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_phi_grad(wheelbase, phi, u1, u2):
    grad = select_grads(tf_phi_new_grad, [tf_phi, tf_u1, tf_u2])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_phi: phi, tf_u1: u1, tf_u2: u2})


# Extra dynamics equations; mostly just rearrangements of the above
v_new_from_px_dot_and_theta_new = tf.div(tf_px_dot, tf.cos(tf_theta_new))
v_new_from_px_dot_and_theta_new_grad = parse_grad(tf.gradients(v_new_from_px_dot_and_theta_new, grad_variables))

v_new_from_py_dot_and_theta_new = tf.div(tf_py_dot, tf.sin(tf_theta_new))
v_new_from_py_dot_and_theta_new_grad = parse_grad(tf.gradients(v_new_from_py_dot_and_theta_new, grad_variables))

v_new_from_theta_dot_phi_new = tf.div(tf.multiply(tf_theta_dot, tf_wheelbase), tf.tan(tf_phi_new))
v_new_from_theta_dot_phi_new_grad = parse_grad(tf.gradients(v_new_from_theta_dot_phi_new, grad_variables))

phi_new_from_theta_dot_v_new = tf.atan(tf.div(tf.multiply(tf_theta_dot, tf_wheelbase), tf_v_new))
phi_new_from_theta_dot_v_new_grad = parse_grad(tf.gradients(phi_new_from_theta_dot_v_new, grad_variables))


def f_x_new_from_x_theta_new_v_new(wheelbase, px, theta, v):
    return sess.run(tf_px_new, feed_dict={tf_wheelbase: wheelbase, tf_px: px, tf_theta_new: theta, tf_v_new: v})

def grad_x_new_from_x_theta_new_v_new(wheelbase, px, theta, v):
    grad = select_grads(tf_px_new_grad, [tf_px, tf_theta, tf_v])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_px: px, tf_theta_new: theta, tf_v_new: v})

def f_y_new_from_y_theta_new_v_new(wheelbase, py, theta, v):
    return sess.run(tf_py_new, feed_dict={tf_wheelbase: wheelbase, tf_py: py, tf_theta_new: theta, tf_v_new: v})

def grad_y_new_from_y_theta_new_v_new(wheelbase, py, theta, v):
    grad = select_grads(tf_py_new_grad, [tf_py, tf_theta, tf_v])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_py: py, tf_theta_new: theta, tf_v_new: v})

def f_theta_new_from_theta_v_new_phi_new(wheelbase, theta, v, phi):
    return sess.run(tf_theta_new, feed_dict={tf_wheelbase: wheelbase, tf_theta: theta, tf_v_new: v, tf_phi_new: phi})

def grad_theta_new_from_theta_v_new_phi_new(wheelbase, theta, v, phi):
    grad = select_grads(tf_theta_new_grad, [tf_theta, tf_v, tf_phi])
    return sess.run(grad, feed_dict={tf_wheelbase: wheelbase, tf_theta: theta, tf_v_new: v, tf_phi_new: phi})

def f_v_new_from_px_dot_and_theta_new(px_dot, theta_new):
    return sess.run(v_new_from_px_dot_and_theta_new, feed_dict={tf_px_dot:px_dot, tf_theta_new:theta_new})

def grad_v_new_from_px_dot_and_theta_new(px_dot, theta_new):
    grad = select_grads(v_new_from_px_dot_and_theta_new_grad, [tf_px_dot, tf_theta_new])
    result = sess.run(grad, feed_dict={tf_px_dot:px_dot, tf_theta_new:theta_new})
    return result

def f_v_new_from_py_dot_and_theta_new(py_dot, theta_new):
    return sess.run(v_new_from_py_dot_and_theta_new, feed_dict={tf_py_dot:py_dot, tf_theta_new:theta_new})

def grad_v_new_from_py_dot_and_theta_new(py_dot, theta_new):
    grad = select_grads(v_new_from_py_dot_and_theta_new_grad, [tf_py_dot, tf_theta_new])
    result = sess.run(grad, feed_dict={tf_py_dot:py_dot, tf_theta_new:theta_new})
    return result

def f_v_new_from_theta_dot_phi_new(wheelbase, theta_dot, phi_new):
    return sess.run(v_new_from_theta_dot_phi_new, feed_dict={tf_wheelbase:wheelbase, tf_theta_dot:theta_dot, tf_phi_new:phi_new})

def grad_v_new_from_theta_dot_phi_new(wheelbase, theta_dot, phi_new):
    grad = select_grads(v_new_from_theta_dot_phi_new_grad, [tf_theta_dot, tf_phi_new])
    result = sess.run(grad, feed_dict={tf_wheelbase:wheelbase, tf_theta_dot:theta_dot, tf_phi_new:phi_new})
    return result

def f_phi_new_from_theta_dot_v_new(wheelbase, theta_dot, v_new):
    return sess.run(phi_new_from_theta_dot_v_new, feed_dict={tf_wheelbase:wheelbase, tf_theta_dot:theta_dot, tf_v_new:v_new})

def grad_phi_new_from_theta_dot_v_new(wheelbase, theta_dot, v_new):
    grad = select_grads(phi_new_from_theta_dot_v_new_grad, [tf_theta_dot, tf_phi_new])
    result = sess.run(grad, feed_dict={tf_wheelbase:wheelbase, tf_theta_dot:theta_dot, tf_v_new:v_new})
    return result
