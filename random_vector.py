import tensorflow as tf
import numpy as np

# This function will yield the randomed binary scaling vector B in Fastfood
# Input:
#  + d: a scalar: size of vector B
# Output:
#  + B: [d, 1]: a randomed binary scaling vector whose diagonal element is sampled from {-1, +1}
def create_binary_scaling_vector(d):
    r_u = tf.random_uniform([1, d], minval=0, maxval=1.0, dtype=tf.float32)
    ones = tf.ones([1, d])
    means = tf.multiply(0.5, ones)
    B = tf.where(r_u > means, ones, tf.multiply(-1.0, ones))
    return tf.reshape(B, [d, 1])

# This function will yield the permutation vector in Fastfood
# Input:
#  + d: a scalar: size of vector pi
# Output:
#  + pi: [d, 1]: a permutation vector of [0, d-1]
def create_permutation(d):
    pi = tf.reshape(tf.range(d), [d, 1])
    pi = tf.random_shuffle(pi)
    return tf.cast(pi, tf.float32)

# This function will yield the gaussian scaling vector G in Fastfood
# Input:
#  + d: a scalar: size of vector G
# Output:
#  + G: [d, 1]: a gaussian vector
def create_gaussian_scaling_vector(d):
    return tf.random_normal([d, 1])

# This function will yield the scaling vector S in Fastfood
# Input:
#  + d: a scalar: size of vector S
#  + G: [d, 1]: the gaussian scaling vector
# Output:
#  + S: [d, 1]: a scaling vector
def create_scaling_vector(d, G):
    frob_norm_G = tf.sqrt(tf.reduce_sum(tf.multiply(G, G)))
    sqrt_frob_norm_G = tf.sqrt(frob_norm_G)
    inv_sqrt_frob_norm_G = tf.div(1.0, sqrt_frob_norm_G)
    s = tf.random_normal([d, 1])
    return tf.multiply(inv_sqrt_frob_norm_G, s)