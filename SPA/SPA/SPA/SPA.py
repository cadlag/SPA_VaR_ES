# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 11:15:30 2017

@author: Daniel
"""

class SPA(object):
    
    def __init__(self, my_dist):
        self.my_dist_ = my_dist
    
    def getSaddlepoint(self, K = None):
        from scipy.optimize import brentq
        from numpy import sign

        guess = 0
        sgn = sign(K - self.my_dist_.CGF(guess, 1))
        func = lambda x : self.my_dist_.CGF(x, 1) - K
        
        if sgn == 0:
            res = guess
        else:
            i = 0
            while sign(K - self.my_dist_.CGF(sgn*2**i, 1)) == sgn:
                i += 1
            res = brentq(func, guess if i == 0 else sgn*2**(i-1), sgn*2**i)

        return res
    
    def approximate(self, K = None, order = 1):
        pass
    
    def __str__(self):
        pass
    
    def getMaxOrder(self):
        import numpy as np
        return np.inf
    
class SPA_LR(SPA):
    
    def __init__(self, MyDist):
        self.my_dist_ = MyDist
        
    def approximate(self, K, order = 1, discrete = False):
        from math import pi
        from scipy.stats import norm
        from numpy import sign, mean, sqrt, exp
        
        p = []
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            return 0.5 - 1.0/(6*sqrt(2*pi))*self.my_dist_.CGF(0,3)/self.my_dist_.CGF(0,2)**1.5
            Ks = [K*.99, K*1.01]
        else:
            Ks = [K]
        for k in Ks:
            sp = self.getSaddlepoint(k)
            #print 'saddlepoint: {}.'.format(sp)
            u = ((1.0 - exp(-sp)) if discrete else sp) * sqrt(self.my_dist_.CGF(sp,2))
            w = sign(sp)*sqrt(2.0*(k*sp-self.my_dist_.CGF(sp,0)))
            p += [1.0-norm.cdf(w)+norm.pdf(w)*(1.0/u-1.0/w )]
            
        return mean(p)
    
    def __str__(self):
        return 'LR_tail_proba'

class SPA_Martin(SPA):

    def __init__(self, my_dist):
        self.my_dist_ = my_dist

    def approximate(self, K = None, order = 1):
        from math import pi
        from scipy.stats import norm
        from numpy import sign, mean, sqrt, exp

        p = []
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            Ks = [K*.99, K*1.01]
        else:
            Ks = [K]
        for k in Ks:
            sp = self.getSaddlepoint(k)
            #print 'saddlepoint: {}.'.format(sp)
            u = sp * sqrt(self.my_dist_.CGF(sp,2))
            w = sign(sp)*sqrt(2.0*(k*sp-self.my_dist_.CGF(sp,0)))
            mu = self.my_dist_.CGF(0, 1)
            approx = mu * (1 - norm.cdf(w)) + norm.pdf(w) * (k / u - mu / w)
            if order > 1:
                cumulant = lambda n: self.my_dist_.CGF(sp, n) / self.my_dist_.CGF(sp, 2)**(n/2.0)
                approx += norm.pdf(w)*(mu/w**3 - k / u**3 - k*cumulant(3) /2.0/u**2 + \
                    k/u*(cumulant(4) / 8.0 - 5.0* cumulant(3)**2 / 24) + 1.0/sp/u)
            p += [approx]

        return mean(p)

    def __str__(self):
        return 'Martin_tail_expectation: E[X1_{X>K}]'


class SPA_Studer(SPA):
    def __init__(self, my_dist):
        self.my_dist_ = my_dist

    def approximate(self, K = None, order = 1):
        from mydistribution import StuderTiltedDist, StuderTiltedDistNeg
        return self.my_dist_.CGF(0, 1) * SPA_LR(StuderTiltedDist(self.my_dist_)).approximate(K)
        #return self.my_dist_.CGF(0, 1) * (1.0 - SPA_LR(StuderTiltedDistNeg(self.my_dist_)).approximate(-K))

    def __str__(self):
        return 'Studer_tail_expectation: E[X1_{X>K}]'

class SPA_ButlerWood(SPA):
    def __init__(self, my_dist):
        return super(SPA_ButlerWood, self).__init__(my_dist)

    def approximate(self, K = None, order = 1):
        from math import pi
        from scipy.stats import norm
        from numpy import sign, mean, sqrt, exp

        p = []
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            Ks = [K*.99, K*1.01]
        else:
            Ks = [K]
        for k in Ks:
            sp = self.getSaddlepoint(k)
            #print 'saddlepoint: {}.'.format(sp)
            u = sp * sqrt(self.my_dist_.CGF(sp,2))
            w = sign(sp)*sqrt(2.0*(k*sp-self.my_dist_.CGF(sp,0)))
            mu = self.my_dist_.CGF(0, 1)
            approx = mu * (1 - norm.cdf(w)) + norm.pdf(w) * (k / u - mu / w)
            if order > 1:
                cumulant = lambda n: self.my_dist_.CGF(sp, n) / self.my_dist_.CGF(sp, 2)**(n/2.0)
                approx += norm.pdf(w)*( (mu - k) / w**3 + 1/sp / u)
            p += [approx]

        return mean(p)

    def __str__(self):
        return 'Huang_tail_expectation: E[X1_{X>K}]'


        