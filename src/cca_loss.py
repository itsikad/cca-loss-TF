import tensorflow as tf
import numpy as np


@tf.custom_gradient
def cca_loss(x1, x2, r1=0.0, r2=0.0):
    """
    An implementation of Deep CCA loss, following the original paper:
    Deep Canonical Correlation Analysis
    Link: https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf

    Inputs: 
        x1: an NxD1 tensor or numpy array, each row correponds to a single sample
        x2: an NxD2 tensor or numpy array, each row correponds to a single sample
        r1: regularizer term for empirical covariance of x1, typically cross-validated
        r2: regularizer term for empirical covariance of x2, typically cross-validated
    Outputs:
        loss: -corr(x1,x2), minus correlation between the x1 and x2
        grad: a sequence of 4 tensors, grad[0/1] - gradient w.r.t x1/2
                                       grad[2/3] - None (gradient w.r.t to r1,r2)
    """

    # Check if numpy array or tensor
    is_ndarray = isinstance(x1, np.ndarray)
    if is_ndarray:
        x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
        x2 = tf.convert_to_tensor(x2, dtype=tf.float32)

    # Extract dimensions
    N = tf.shape(x1)[0]
    N_float = tf.cast(N, 'float')
    D1 = tf.shape(x1)[1]
    D2 = tf.shape(x2)[1]

    # Compute scaling factors
    scale_factor = tf.divide(1., N_float-1.)
    scale_mat = tf.eye(N) - tf.divide(1., N_float) * tf.ones([N, N])

    # Compute h_bar matrices (note that the transpose matrices are used)
    h1_bar = tf.matmul(x1, scale_mat, adjoint_a=True)
    h2_bar = tf.matmul(x2, scale_mat, adjoint_a=True)

    # Compute covariance matrices
    cov_11 = scale_factor * tf.matmul(h1_bar, h1_bar, adjoint_b=True) + r1 * tf.eye(D1)
    cov_22 = scale_factor * tf.matmul(h2_bar, h2_bar, adjoint_b=True) + r2 * tf.eye(D2)
    cov_12 = scale_factor * tf.matmul(h1_bar, h2_bar, adjoint_b=True)

    # Compute R and its SVD decomposition
    cov_11_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(cov_11))
    cov_22_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(cov_22))
    R = tf.matmul(tf.matmul(cov_11_sqrt_inv, cov_12), cov_22_sqrt_inv)
    s, u, v = tf.linalg.svd(R)

    # Compute loss
    loss = tf.multiply(tf.constant(-1.0), tf.reduce_sum(s))
    # loss = -tf.linalg.trace(tf.linalg.sqrtm(tf.matmul(tf.transpose(R), R)))

    # Compute gradients
    def gradients(dy):
        cov_11_sqrt_inv_u = tf.matmul(cov_11_sqrt_inv, u)
        cov_22_sqrt_inv_v = tf.matmul(cov_22_sqrt_inv, v)
        delta_11 = tf.matmul(-0.5 * cov_11_sqrt_inv_u, tf.matmul(tf.linalg.diag(s), cov_11_sqrt_inv_u, adjoint_b=True))
        delta_22 = tf.matmul(-0.5 * cov_22_sqrt_inv_v, tf.matmul(tf.linalg.diag(s), cov_22_sqrt_inv_v, adjoint_b=True))
        delta_12 = tf.matmul(tf.matmul(cov_11_sqrt_inv, u), tf.matmul(v, cov_22_sqrt_inv, adjoint_a=True))

        # Calculate final gradient, multiply by minus downstream gradient since -corr is used as loss
        grad_h1 = -scale_factor * (tf.matmul(2. * delta_11, h1_bar) + tf.matmul(delta_12, h2_bar))
        grad_h2 = -scale_factor * (tf.matmul(2. * delta_22, h2_bar) + tf.matmul(delta_12, h1_bar, adjoint_a=True))

        # Transpose back
        grad_h1_t = tf.transpose(grad_h1)
        grad_h2_t = tf.transpose(grad_h2)

        # Note: Return None for gradients w.r.t r1 and r2
        return grad_h1_t, grad_h2_t

    return loss, gradients
    # return loss
