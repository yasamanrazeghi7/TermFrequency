import numpy as np
import pymc3 as pm
import theano.tensor as tt

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))