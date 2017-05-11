import numpy as np
import scipy

# Some Variable declarations:
datapoints=int(1e+3)
kgaussian=3

#Function for getting the covaraince matrix
def covmatrix(var,covar):
    """Will be used for constructing the covariance matrix required.
    param variances: Vector of variances for normal distributions
    param covariances: Vector of covariances, here first value is Cov12, second Cov23, Third is Cov31
    """
    cov=np.zeros((kgaussian,kgaussian))
    cov[0,0]=var[0]
    cov[1,1]=var[1]
    cov[2,2]=var[2]
    cov[0,1]=cov[1,0]=covar[0]
    cov[0,2]=cov[2,0]=covar[2]
    cov[1,2]=cov[2,1]=covar[1]
    return cov

#Function for first initialization of stuff
def initialize():
    means=np.random.rand(1,kgaussian)
    var=np.random.rand(1,kgaussian)
    covar=np.random.rand(1,kgaussian)
    cov=covmatrix(var,covar)
    mixingprob=np.random.random_sample(3)
    summix=np.sum(mixingprob)
    mixingprob=np.divide(mixingprob,summix)
    return means,var,cov,mixingprob

def loglike(data,means,var,cov,mixingprob):
    """ Computes the log likelihood
    param data: is the set of data points that we have, formally called incomplete data
    param means, var, mixingprob: Respectively the means, variances and the mixing probabilties that we have
    param cov: The covariance matrix, of dimensions kgaussian X kgaussian

