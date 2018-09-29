# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 11:15:30 2017

@author: Daniel
"""

from mydistribution import *

class SPA(object):
    
    def __init__(self, my_dist):
        self.my_dist_ = my_dist
        self.spCache_ = dict()
    
    def getSaddlepoint(self, K = None):
        if not K in self.spCache_.keys():
            from scipy.optimize import brentq
            from numpy import sign

            guess = 0
            sgn = sign(K - self.my_dist_.CGF(self.my_dist_.transform(guess), 1))
            func = lambda x : self.my_dist_.CGF(self.my_dist_.transform(x), 1) - K
        
            if sgn == 0:
                res = guess
            else:
                i = 0
                while sign(K - self.my_dist_.CGF(self.my_dist_.transform(sgn*2**i), 1)) == sgn:
                    i += 1
                res = brentq(func, guess if i == 0 else sgn*2**(i-1), sgn*2**i)
            self.spCache_[K] = self.my_dist_.transform(res)
        return self.spCache_[K]
    
    def approximate(self, K = None, order = 1):
        pass
    
    def __str__(self):
        pass
    
    def getMaxOrder(self):
        import numpy as np
        return np.inf
    
class SPA_LR(SPA):
    
    def __init__(self, myDist):
        return super(SPA_LR, self).__init__(myDist)
        
    def approximate(self, K, order = 1, discrete = False):
        from math import pi
        from scipy.stats import norm
        from numpy import sign, mean, sqrt, exp
        
        p = []
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            #return 0.5 - 1.0/(6*sqrt(2*pi))*self.my_dist_.CGF(0,3)/self.my_dist_.CGF(0,2)**1.5
            Ks = [K*.999, K*1.001]
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

    def __init__(self, myDist):
        return super(SPA_Martin, self).__init__(myDist)

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
    def __init__(self, myDist):
        return super(SPA_Studer, self).__init__(myDist)

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
            Ks = [K*.999, K*1.001]
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

class SPANonGaussian(SPA):
    def __init__(self, myDist, baseDist = None):
        super(SPANonGaussian, self).__init__(myDist)
        self.baseDist_ = baseDist
        self.fittedBaseDists_ = dict()

    def getSaddlepoint2(self, K):
        from scipy.optimize import brentq
        from numpy import sign

        z_hat = self.getSaddlepoint(K)
        baseDist = self.getBaseDist(K)

        guess = 0
        rhs = self.my_dist_.CGF(z_hat, 0) - z_hat * K
        sgn = sign(rhs - baseDist.CGF(baseDist.transform(guess), 0) + baseDist.transform(guess) * baseDist.CGF(baseDist.transform(guess), 1));
        func = lambda x : baseDist.CGF(baseDist.transform(x), 0) - baseDist.transform(x) * baseDist.CGF(baseDist.transform(x), 1) - rhs
        
        if sgn == 0:
            res = guess
        else:
            i = 0
            sgn1 = sign(K - self.my_dist_.CGF(0, 1))
            while sign(-func(sgn1*2**i)) == sgn and i < 50:
                i += 1
            res = brentq(func, guess if i == 0 else sgn1*2**(i-1), sgn1*2**i)

        return baseDist.transform(res)

    def getBaseDist(self, K = None):
        if isinstance(self.baseDist_, MyDistribution):
            return self.baseDist_
        if isinstance(self.baseDist_, str) and K != None:
            if not K in self.fittedBaseDists_.keys():
                if self.baseDist_.lower() == "gamma":
                    z_h = self.getSaddlepoint(K)
                    self.fittedBaseDists_[K] = MyGamma(6*self.my_dist_.CGF(z_h, 2)**2/self.my_dist_.CGF(z_h, 4), 1.0)
                elif self.baseDist_.lower() == "invgauss":
                    from numpy import sign, sqrt
                    z_h = self.getSaddlepoint(K)
                    xi_4 = self.my_dist_.CGF(z_h, 4) / self.my_dist_.CGF(z_h, 2)**2
                    c = z_h * K - self.my_dist_.CGF(z_h, 0)
                    y = c*sign(z_h) + sqrt(2*c**2 + 30.0*c/xi_4)
                    lam = (y**2 - c**2)/2.0/c if c != 0.0 else 15.0 / xi_4
                    self.fittedBaseDists_[K] = MyInvGauss(lam, 1.0)
                else:
                    raise Exception("base distribution type " + self.baseDist_ + " not supported.")
            return self.fittedBaseDists_[K]

        return MyNormal()

class SPANonGaussian_Wood(SPANonGaussian):
    def __init__(self, myDist, baseDist):
        return super(SPANonGaussian_Wood, self).__init__(myDist, baseDist)

    def approximate(self, K = None, order = 1):
        from numpy import mean
        p = []
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            vK = [K*.999, K*1.001]
        else:
            vK = [K]
        for k in vK:
            baseDist = self.getBaseDist(k)
            z_h = self.getSaddlepoint(k)
            w_h = self.getSaddlepoint2(k)
            k0_1 = baseDist.CGF(w_h, 1)
            k0_2 = baseDist.CGF(w_h, 2)
            k_2 = self.my_dist_.CGF(z_h, 2)
            p += [1.0 - baseDist.cdf(k0_1) - baseDist.density(k0_1) * (
            1.0 / w_h - 1.0/z_h * (k0_2 / k_2)**0.5)]
        return mean(p)

    def __str__(self):
        return "Wood_tail_probability: P(X>K)"

class SPANonGaussian_ZK(SPANonGaussian):
    def __init__(self, myDist, baseDist):
        return super(SPANonGaussian_ZK, self).__init__(myDist, baseDist)

    def approximate(self, K = None, order = 1):
        from numpy import sqrt, mean
        p = []
        baseDist = self.getBaseDist(K)
        wood = SPANonGaussian_Wood(self.my_dist_, baseDist)
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            vK = [K*.999, K*1.001]
        else:
            vK = [K]
        for k in vK:
            res = 0.0
            z_h = self.getSaddlepoint(k)
            w_h = self.getSaddlepoint2(k)
            k_1_0 = self.my_dist_.CGF(0, 1)
            k0_1 = baseDist.CGF(w_h, 1)
            k0_2 = baseDist.CGF(w_h, 2)
            k0_3 = baseDist.CGF(w_h, 3)
            k_2 = self.my_dist_.CGF(z_h, 2)
            mu_h = z_h * sqrt(k_2)
            dx = max(0.0001, k0_1*0.0001)
            res += k_1_0*wood.approximate(k) + baseDist.density(k0_1) * ((k-k_1_0)*(1/w_h - 1/w_h**3/k0_2 - k0_3/2/w_h/k0_2**1.5/mu_h) + sqrt(k0_2)/mu_h/z_h)
            res += (baseDist.density(k0_1 + dx) - baseDist.density(k0_1 - dx))/2/dx * (k - k_1_0)* (1/w_h**2 - sqrt(k0_2) / w_h/mu_h)
            p += [res]
        return mean(p)

    def __str__(self):
        return "Zhang_ZK_tail_expectaion"
       
class SPANonGaussian_HO(SPANonGaussian):
    def __init__(self, myDist, baseDist):
        return super(SPANonGaussian_HO, self).__init__(myDist, baseDist)

    def approximate(self, K = None, order = 1):
        from numpy import sqrt, mean
        p = []
        baseDist = self.getBaseDist(K)
        wood = SPANonGaussian_Wood(self.my_dist_, baseDist)
        if abs(K - self.my_dist_.CGF(0,1)) < 1e-6:
            vK = [K*.999, K*1.001]
        else:
            vK = [K]
        for k in vK:
            z_h = self.getSaddlepoint(k)
            w_h = self.getSaddlepoint2(k)
            k_1_0 = self.my_dist_.CGF(0, 1)
            k0_1 = baseDist.CGF(w_h, 1)
            k0_2 = baseDist.CGF(w_h, 2)
            #k0_3 = baseDist.CGF(w_h, 3)
            k_2 = self.my_dist_.CGF(z_h, 2)
            k_1 = self.my_dist_.CGF(z_h, 1)
            nu_h = (k_1 - k_1_0)/(k0_1 - baseDist.CGF(0, 1))
            res = k*wood.approximate(k) + baseDist.density(k0_1) * (sqrt(k0_2/k_2)/z_h**2 - nu_h/w_h**2)
            res += baseDist.tail_expectation(k0_1) * nu_h
            p += [res]
        return mean(p)

    def __str__(self):
        return "Zhang_HO_tail_expectaion" 