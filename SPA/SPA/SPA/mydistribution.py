# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 15:17:40 2017

@author: Daniel
"""

class MyDistribution(object):
    
    def __init__(self):
        pass
    
    def CGF(self, x, order = 0):
        pass
    
    def getMaxOrder(self):
        from numpy import inf
        return inf

    def copy(self):
        from copy import copy
        return copy(self)

    def transform(self, x):
        return x
        

class MyNormal(MyDistribution):
    
    def __init__(self, mean = 0, std = 1):       
        self.mean_ = mean
        self.sigma_ = std
            
    def CGF(self, x, order = 0):
        if order == 0:
            return self.mean_*x + 1.0/2 * self.sigma_**2 * x**2
        elif order == 1:
            return self.mean_ + self.sigma_**2*x
        elif order == 2:
            return self.sigma_**2
        else:
            return 0.0

    def density(self, x):
        from scipy.stats import norm
        return norm.pdf(x, loc = self.mean_, scale = self.sigma_)

    def cdf(self, x):
        from scipy.stats import norm
        return norm.cdf(x, loc = self.mean_, scale = self.sigma_)

    def ppf(self, x):
        from scipy.stats import norm
        return norm.ppf(x, loc = self.mean_, scale = self.sigma_)

    def rvs(self, size = 1, seed = None):
        from scipy.stats import norm
        return norm.rvs(loc = self.mean_, scale = self.sigma_, size = size)

    def tail_expectation(self, x):
        import numpy as np
        from math import pi
        return self.density(x) - x * self.cdf(-x)
        #return self.sigma_ / 2 / pi * np.exp(- (x - self.mean_)**2/2/self.sigma_**2) + (
        #    self.mean_ - x) * (1 - self.cdf( (x-self.mean_)/self.sigma_))

    def transform(self, x):
        return x



class ConditionalLossDist(MyDistribution):

    def __init__(self, weights, probs, corrs, y_value = 0):

        n = len(weights)
        assert(n == len(probs))
        assert(n == len(corrs))

        self.y_value_ = y_value
        self.weights_ = weights
        self.probs_ = probs
        self.corrs_ = corrs

    def setY(self, y_value, inplace = False):
        if inplace:
            cpy = self
        else:
            cpy = self.copy()
        cpy.y_value_ = y_value
        if not inplace:
            return cpy

    def getMaxOrder(self):
        return 4

    def conditionalDP(self, p, corr):
        from scipy.stats import norm
        from numpy import sqrt
        return norm.cdf( (norm.ppf(p) - sqrt(corr)*self.y_value_) / sqrt(1-corr) )

    #def CGF(self, x, order = 0):

    #    assert(order <= self.getMaxOrder())

    #    import numpy as np
    #    from math import log, exp

    #    summand = 0.0
    #    for i in np.arange(0, len(self.weights_)):
    #        norm_weights = self.weights_ / sum(self.weights_)
    #        cond_p = self.conditionalDP(self.probs_[i], self.corrs_[i])
    #        tmp = 1.0 - cond_p + cond_p * exp(norm_weights[i] * x)
    #        if order == 0:
    #            summand += log(tmp)
    #        elif order == 1:
    #            summand += norm_weights[i] * cond_p * exp(norm_weights[i] * x) / tmp
    #        elif order == 2:
    #            summand += (1- cond_p)*norm_weights[i]**2 * cond_p * exp(norm_weights[i] * x) / tmp**2
    #        elif order == 3:
    #            tmp2 = (1- cond_p)*norm_weights[i]**2 * cond_p * exp(norm_weights[i] * x)
    #            summand += tmp2 * norm_weights[i] / tmp**2 - 2*tmp2 * norm_weights[i]*cond_p * exp(norm_weights[i] * x) / tmp**3
    #        elif order == 4:
    #            tmp2 = (1- cond_p)*norm_weights[i]**4 * cond_p * exp(norm_weights[i] * x)
    #            summand += tmp2/tmp**2 - 6*tmp2*cond_p * exp(norm_weights[i]*x) / tmp**3  
    #            + 6*tmp2 * cond_p**2 ** exp(norm_weights[i]* x * 2) / tmp**4
    #        else:
    #            raise ValueError('invalid order value {}'.format(order))

    #    return summand

    def CGF(self, x, order = 0):

        assert(order <= self.getMaxOrder())

        import numpy as np
        from numpy import log, exp

        norm_weights = self.weights_ / sum(self.weights_)
        cond_p = self.conditionalDP(self.probs_, self.corrs_)
        tmp = 1.0 - cond_p + cond_p * exp(norm_weights * x)
        if order == 0:
            return np.sum(log(tmp))
        elif order == 1:
            return np.sum(norm_weights * cond_p * exp(norm_weights * x) / tmp)
        elif order == 2:
            return np.sum((1- cond_p)*norm_weights**2 * cond_p * exp(norm_weights * x) / tmp**2)
        elif order == 3:
            tmp2 = (1- cond_p)*norm_weights**2 * cond_p * exp(norm_weights * x)
            return np.sum(tmp2 * norm_weights / tmp**2 - 2*tmp2 * norm_weights*cond_p * exp(norm_weights * x) / tmp**3)
        elif order == 4:
            tmp2 = (1- cond_p)*norm_weights**4 * cond_p * exp(norm_weights * x)
            return np.sum(tmp2/tmp**2 - 6*tmp2*cond_p * exp(norm_weights*x) / tmp**3 \
                + 6*tmp2 * cond_p**2 ** exp(norm_weights* x * 2) / tmp**4)
        else:
            raise ValueError('invalid order value {}'.format(order))

class StuderTiltedDist(MyDistribution):

    def __init__(self, inner_dist):
        self.inner_dist_ = inner_dist

    def getMaxOrder(self):
        return super(StuderTiltedDist, self).getMaxOrder() - 1

    def CGF(self, x, order = 0):
        from scipy.misc import comb
        from math import log
        if order <= self.getMaxOrder():
            if order == 0:
                return self.inner_dist_.CGF(x, 0) + log(self.inner_dist_.CGF(x, 1)) - \
                    log(self.inner_dist_.CGF(0, 1))
            elif order == 1:
                return self.inner_dist_.CGF(x, 1) + self.inner_dist_.CGF(x, 2) / self.inner_dist_.CGF(x, 1)
            elif order == 2:
                return self.inner_dist_.CGF(x, 2) + self.inner_dist_.CGF(x, 3) / self.inner_dist_.CGF(x, 1) - \
                    (self.inner_dist_.CGF(x, 2)/self.inner_dist_.CGF(x, 1))**2
            elif order == 3:
                return self.inner_dist_.CGF(x, 3) + self.inner_dist_.CGF(x, 4) / self.inner_dist_.CGF(x, 1) - \
                    self.inner_dist_.CGF(x, 3)*self.inner_dist_.CGF(x, 2) / self.inner_dist_.CGF(x, 1)**2 + \
                    2*self.inner_dist_.CGF(x, 2)/self.inner_dist_.CGF(x, 1) * (self.inner_dist_.CGF(x, 3) / self.inner_dist_.CGF(x, 1) - \
                    (self.inner_dist_.CGF(x, 2)/self.inner_dist_.CGF(x, 1))**2)
               
class StuderTiltedDistNeg(StuderTiltedDist):

    def __init__(self, inner_dist):
        return super(StuderTiltedDistNeg, self).__init__(inner_dist)

    def getMaxOrder(self):
        return super(StuderTiltedDistNeg, self).getMaxOrder() - 1

    def CGF(self, x, order = 0):
        return super(StuderTiltedDistNeg, self).CGF(-x, order)
       

class MyGamma(MyDistribution):
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def density(self, x):
        from scipy.stats import gamma
        return gamma.pdf(x, self.shape, loc=0, scale=self.scale)

    def cdf(self, x):
        from scipy.stats import gamma
        return gamma.cdf(x, self.shape, loc=0, scale=self.scale)

    def CGF(self, x, order = 0):
        from numpy import log
        from math import factorial
        if order == 0:
            return -self.shape * log(1-self.scale * x)
        else:
            return self.shape * self.scale**order / (1-self.scale*x)**order * factorial(order - 1)

    def tail_expectation(self, x): # E[X1_{X>K}]
        gma = MyGamma(self.shape + 1, self.scale)
        return self.shape*self.scale*( 1.0 - gma.cdf(x) )

    def transform(self, x):
        from numpy import exp
        return 1/self.scale*(1.0 - exp(-x))


class MyInvGauss(MyDistribution):
    def __init__(self, shape, scale): # lambda, mu
        self.shape = scale / shape
        self.scale = shape

    def density(self, x):
        from scipy.stats import invgauss
        return invgauss.pdf(x, self.shape, loc=0, scale=self.scale)

    def cdf(self, x):
        from scipy.stats import invgauss
        return invgauss.cdf(x, self.shape, loc=0, scale=self.scale)

    def CGF(self, x, order = 0):
        from numpy import log, sqrt
        if order == 0:
            return self.shape/self.scale * (1- sqrt(1-2*self.scale**2*x/self.shape))
        else:
            f = lambda x, n: x if n <= 0 else f(x, n - 1)*(x-n)
            return self.shape/self.scale * (-2*self.scale**2/self.shape)**order * (1-2*self.scale**2/self.shape)*(0.5 - order) * f(0.5, order - 1)

    def tail_expectation(self, x): # E[X1_{X>K}]
        from scipy.stats import norm
        from numpy import sqrt, exp
        return self.scale*(norm.cdf(-sqrt(self.shape/x)*(x/self.scale - 1)) + exp(2*self.shape/self.scale) * norm.cdf(-sqrt(self.shape/x)*(x/self.scale + 1)))

    def transform(self, x):
        from numpy import exp
        return 1/(2*self.scale**2/self.shape)*(1.0 - exp(-x)) 