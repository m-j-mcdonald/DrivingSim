'''
Definitions to perform dynamics calculations
'''
import tensorflow as tf

from internal_state.constants import time_delta

tf_px = tf.Variable(0, 'px')
tf_py = tf.Variable(0, 'py')
tf_theta = tf.Variable(0, 'theta')
tf_v = tf.Variable(0, 'theta')
tf_phi = tf.Variable(0, 'theta')
tf_wheelbase = tf.Variable(0, 'wheelbase')
tf_time_delta = tf.Constant(time_delta, 'time_delta')

tf_u1 = tf.Variable(0, 'u1') # v_dot
tf_u2 = tf.Variable(0, 'u2') # phi_dot
tf_v_new = tf.add(tf_v, tf.mult(tf_time_delta, tf_u1))
tf_phi_new = tf.add(tf_phi, tf.mult(tf_time_delta, tf_u2))

px_dot = tf.mult(tf_v_new, tf.cos(tf.theta))
py_dot = tf.mult(tf_v_new, tf.sin(tf.theta))
theta_dot = tf.mult(tf_v_new, tf.div(tf.tan(tf_phi), tf_wheelbase))

tf_px_new = tf.add(tf_px, tf.mult(tf_time_delta, px_dot))
tf_py_new = tf.add(tf_py, tf.mult(tf_time_delta, py_dot))
tf_theta_new = tf.add(tf_theta, tf.mult(tf_time_delta, theta_dot))

tf_px_new_grad = tf.gradients(tf_px_new, [tf_u1, tf_u2])
tf_py_new_grad = tf.gradients(tf_py_new, [tf_u1, tf_u2])
tf_theta_new_grad = tf.gradients(tf_theta_new, [tf_u1, tf_u2])
tf_v_new_grad = tf.gradients(tf_v_new, [tf_u1, tf_u2])
tf_phi_new_grad = tf.gradients(tf_phi_new, [tf_u1, tf_u2])

def run_equations(px, py, theta, v, phi, wheelbase, u1, u2, sess):
    return sess.run([tf_px_new, tf_py_new, tf_theta, tf_v_new, tf_phi_new], feed_dict={tf_px: px,
                                                                                       tf_py: py,
                                                                                       tf_theta: theta,
                                                                                       tf_v: v,
                                                                                       tf_phi: phi,
                                                                                       tf_wheelbase: wheelbase,
                                                                                       tf_u1: u1,
                                                                                       tf_u2: u2})

def next_px_f(px, theta, v, u1, u2, sess):
    return sess.run(tf_px_new, feed_dict={tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_px_grad(px, theta, v, u1, u2, sess):
    return sess.run(tf_px_new_grad, feed_dict={tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_f(py, theta, v, u1, u2, sess):
    return sess.run(tf_py_new, feed_dict={tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_grad(py, theta, v, u1, u2, sess):
    return sess.run(tf_py_new_grad, feed_dict={tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_theta_f(theta, v, phi, u1, u2, sess):
    return sess.run(tf_theta_new, feed_dict={tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_theta_grad(theta, v, phi, u1, u2, sess):
    return sess.run(tf_theta_new_grad, feed_dict={tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_v_f(v, u1, u2, sess):
    return sess.run(tf_v_new, feed_dict={tf_v: v, tf_u1: u1, tf_u2: u2})

def next_v_grad(v, u1, u2, sess):
    return sess.run(tf_v_new_grad, feed_dict={tf_v: v, tf_u1: u1, tf_u2: u2})

def next_phi_f(phi, u1, u2, sess):
    return sess.run(tf_phi_new, feed_dict={tf_phi: phi, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_phi_grad(phi, u1, u2, sess):
    return sess.run(tf_phi_new_grad, feed_dict={tf_phi: phi, tf_v: v, tf_u1: u1, tf_u2: u2})