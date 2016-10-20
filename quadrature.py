import numpy as np
from numpy.polynomial import hermite


def loss(likelihood, mu, v, Y):
    """
    calculates the negative log likelihood for both the quadrature
    approximation of the marginal likelihood and the closed form approximation

    mu     - vector of mean predictions corresponding to rows in test_X
    v      - vector of variance predictions corresponding to rows in test_X
    test_Y - vector of target values for test set
    """
    n = float(test_Y.shape[0])
    quad_sum, closed_sum = 0.0, 0.0
    for m, var, x, y in zip(mu, v, Y):
        quad_sum -= gauss_hermite_quadrature(likelihood, m, var, y, 20)
        closed_sum -= closed_form(likelihood, m, v)
    return (quad_sum/n, closed_sum/n)


def gauss_hermite_quadrature(likelihood, mu, v, y_star, n):
    """
    uses quadrature to approximate the log predictive distribution
    at the given point

    likelihood - likelihood object
    mu         - mean
    v          - variance
    y_star     - class label
    n          - number of sample points to use in quadrature approximation

    computes:
    log(1/sqrt(pi) * sum(w_i * L(sqrt(2)*sigma*x_i + mu)))
    where w_i = 2^(n-1)n!sqrt(pi) / n^2[H_{n-1}(x_i)]^2
    """
    y_pred_prob = 0.0
    sample_points, weights = hermite.hermgauss(n)
    for p, w in zip(sample_points, weights):
        z = np.sqrt(2.0*v) * p + mu
        # pdf takes into account the link function
        y_pred_prob += likelihood.pdf(z, y_star) * w
    return np.log(y_pred_prob)-np.log(np.sqrt(np.pi))


def closed_form(likelihood, mu, v):
    """
    returns the approximate log predicitve distribution for mu and v
    using a closed form solution given below

    likelihood - likelihood object
    mu         - mean
    v          - variance

    computes:
    log(link((1 + pi*v/8)^(-1/2))*mu)
    """
    return np.log(likelihood.gp_link.transf(pow(1 + (np.pi*v/8.0), -0.5)*mu))
