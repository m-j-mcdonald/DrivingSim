'''
Definitions to perform dynamics calculations
'''
import tensorflow as tf

from driving_utils.constants import time_delta

init_op = tf.global_variables_initializer()

tf_px = tf.Variable(0, 'px', dtype='float32')
tf_py = tf.Variable(0, 'py', dtype='float32')
tf_theta = tf.Variable(0, 'theta', dtype='float32')
tf_v = tf.Variable(0, 'theta', dtype='float32')
tf_phi = tf.Variable(0, 'theta', dtype='float32')
tf_wheelbase = tf.Variable(0, 'wheelbase', dtype='float32')
tf_time_delta = tf.constant(time_delta, dtype='float32', name='time_delta')

tf_u1 = tf.Variable(0, 'u1', dtype='float32') # v_dot
tf_u2 = tf.Variable(0, 'u2', dtype='float32') # phi_dot
tf_v_new = tf.add(tf_v, tf.multiply(tf_time_delta, tf_u1))
tf_phi_new = tf.add(tf_phi, tf.multiply(tf_time_delta, tf_u2))

theta_dot = tf.multiply(tf_v, tf.div(tf.tan(tf_phi), tf_wheelbase))
tf_theta_new = tf.add(tf_theta, tf.multiply(tf_time_delta, theta_dot))

px_dot = tf.multiply(tf_v, tf.cos(tf_theta))
py_dot = tf.multiply(tf_v, tf.sin(tf_theta))
tf_px_new = tf.add(tf_px, tf.multiply(tf_time_delta, px_dot))
tf_py_new = tf.add(tf_py, tf.multiply(tf_time_delta, py_dot))

tf_px_new_grad = tf.gradients(tf_px_new, [tf_u1, tf_u2])
tf_py_new_grad = tf.gradients(tf_py_new, [tf_u1, tf_u2])
tf_theta_new_grad = tf.gradients(tf_theta_new, [tf_u1, tf_u2])
tf_v_new_grad = tf.gradients(tf_v_new, [tf_u1, tf_u2])
tf_phi_new_grad = tf.gradients(tf_phi_new, [tf_u1, tf_u2])

def run_equations(px, py, theta, v, phi, wheelbase, u1, u2, sess):
    sess.run(init_op)
    return sess.run([tf_px_new, tf_py_new, tf_theta, tf_v_new, tf_phi_new], feed_dict={tf_px: px,
                                                                                       tf_py: py,
                                                                                       tf_theta: theta,
                                                                                       tf_v: v,
                                                                                       tf_phi: phi,
                                                                                       tf_wheelbase: wheelbase,
                                                                                       tf_u1: u1,
                                                                                       tf_u2: u2})

def next_px_f(px, theta, v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_px_new, feed_dict={tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_px_grad(px, theta, v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_px_new_grad, feed_dict={tf_px: px, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_f(py, theta, v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_py_new, feed_dict={tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_py_grad(py, theta, v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_py_new_grad, feed_dict={tf_py: py, tf_theta: theta, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_theta_f(theta, v, phi, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_theta_new, feed_dict={tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_theta_grad(theta, v, phi, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_theta_new_grad, feed_dict={tf_theta: theta, tf_v: v, tf_phi: phi, tf_u1: u1, tf_u2: u2})

def next_v_f(v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_v_new, feed_dict={tf_v: v, tf_u1: u1, tf_u2: u2})

def next_v_grad(v, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_v_new_grad, feed_dict={tf_v: v, tf_u1: u1, tf_u2: u2})

def next_phi_f(phi, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_phi_new, feed_dict={tf_phi: phi, tf_v: v, tf_u1: u1, tf_u2: u2})

def next_phi_grad(phi, u1, u2, sess):
    sess.run(init_op)
    return sess.run(tf_phi_new_grad, feed_dict={tf_phi: phi, tf_v: v, tf_u1: u1, tf_u2: u2})
