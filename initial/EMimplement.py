import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import scipy.stats
# Some Variable declarations:
datapoints=999
kgaussian=3
epoch=10
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
def initialize():
    means=np.random.rand(kgaussian)
    # print means.shape
    var=np.random.rand(kgaussian)
    # print var.shape
    covar=np.random.rand(kgaussian)
    cov=covmatrix(var,covar)
    mixingprob=np.random.random_sample(3)
    summix=np.sum(mixingprob)
    mixingprob=np.divide(mixingprob,summix)
    # print mixingprob
    return means,var,cov,mixingprob
def evalloglike(data,means,var,cov,mixingprob):
    """ Computes the log likelihood
    param data: is the set of data points that we have, formally called incomplete data
    param means, var, mixingprob: Respectively the means, variances and the mixing probabilties that we have
    param cov: The covariance matrix, of dimensions kgaussian X kgaussian
    """
    # P(X|u,E,pie) need to be calculated. Actually log of that.
    sum=0;
    for n in range(0,datapoints):
        pdfvector=np.zeros((3))
        pdfvector[0]=scipy.stats.norm(means[0],var[0]).cdf(data[n])
        #since my data points are 1-D, I can use single variate mean and variance. Else I would have used multivariate normal distribution, with each distribution having its own mean vector and covariance matrix, which indicates the mean of its Dimensions, and cov between dimensions
        pdfvector[1]=scipy.stats.norm(means[1],var[1]).cdf(data[n])
        pdfvector[2]=scipy.stats.norm(means[2],var[2]).cdf(data[n])
        inner=np.dot(mixingprob,pdfvector)
        # print "Inner: ",inner
        sum=sum+np.log(inner)
    print sum
    return sum
def posterior(n,k,data,means,var,mixingprob):
    """Evaluate the responsibilities
    """
    #When this function is to be called, make sure that n and k are 0 indexed
    # print "Printing k value in posterior function: ",k
    numerator=mixingprob[k]*scipy.stats.norm(means[k],var[k]).cdf(data[n])
    denom=0
    for i in range(0,kgaussian):
        denom=denom+mixingprob[i]*scipy.stats.norm(means[i],var[i]).cdf(data[n])
    value=numerator/denom
    # print value
    return value
def calcposterior(k,data,means,var,mixingprob):
    gammank_vector=np.zeros((k,datapoints))
    for j in range(0,k):
        for i in range(0,datapoints):
            gammank_vector[j,i]=posterior(i,j,data,means,var,mixingprob)
    return gammank_vector
def Nk(k,data,means,var,mixingprob,gammavector):
    """Calculates N_k required for updates"""
    return np.sum(gammavector[k,:])
def maximizestep(k,data,means,var,mixingprob,gammank_vector):
    """We'll re-estimate the parameters, using current posterior probabilties.
    Maximizes only the kth parameter. Make sure k and n are 0 indexed
    We have k Guassian probabilites.
    """
    nk=Nk(k,data,means,var,mixingprob,gammank_vector)
    print "------"
    print "Printing means vector before updating "
    print means
    means[k]=np.dot(gammank_vector[k,:],data)/nk
    print "Printing means vector after updating ",k,"Gaussian"
    print means
    print "------"
    numerator=0
    for i in range(0,datapoints):
        numerator=numerator+gammank_vector[k,i]*np.square(data[i]-means[k])
    var[k]=numerator/nk
    mixingprob[k]=nk/datapoints
    return means,var,mixingprob
def ExpectMax():
    means,var,cov,mixingprob=initialize()
    means=[5,20,50]
    var=[2,2,2]
    mixingprob=[0.3333,0.3333,0.3333]
    print means
    print var
    data=np.random.normal(5,2,datapoints/3)
    data2=np.random.normal(20,2,datapoints/3)
    data3=np.random.normal(50,2,datapoints/3)
    plt.plot(data,color="r")
    plt.plot(data2,color="b")
    plt.plot(data3,color="g")
    plt.show()
    # print data1
    data=np.append(data,data2)
    data=np.append(data,data3)
    np.random.shuffle(data)
    print "Shape check of data: ",data.shape
    likelihood=np.zeros((epoch))
    likelihood[0]=evalloglike(data,means,var,cov,mixingprob)
    gammank_vector=calcposterior(kgaussian,data,means,var,mixingprob)
    for iterator in range(1,epoch):
        for i in range(0,kgaussian):
            means,var,mixingprob=maximizestep(i,data,means,var,mixingprob,gammank_vector)
        #Done updating[Maximizing for all the K guassian parameters]
        likelihood[iterator]=evalloglike(data,means,var,cov,mixingprob)
        gammank_vector=calcposterior(kgaussian,data,means,var,mixingprob)
    print "The obtained means,var and mixing prob"
    print means
    print var
    print mixingprob
    ##Plotting of loglikelihood vs Epoch
    plt.plot(likelihood)
    plt.title("Likelihood vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Log Likelihood value")
    plt.show()

ExpectMax()
