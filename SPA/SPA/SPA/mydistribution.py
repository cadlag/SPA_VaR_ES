# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 15:17:40 2017

@author: Daniel
"""

import sympy as sym

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
        from scipy.stats import norm
        return self.sigma_*norm.pdf((x-self.mean_)/self.sigma_) + self.mean_ * norm.cdf(-(x-self.mean_)/self.sigma_) - x*(1-self.cdf(x))
        #return self.sigma_ / 2 / pi * np.exp(- (x - self.mean_)**2/2/self.sigma_**2) + (
        #    self.mean_ - x) * (1 - self.cdf( (x-self.mean_)/self.sigma_))

    def transform(self, x):
        return x


    # uniform correlation
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
        if x <= 0:
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
                    + 6*tmp2 * cond_p**2 * exp(norm_weights* x * 2) / tmp**4)
            else:
                raise ValueError('invalid order value {}'.format(order))
        else:
            tmp = (1.0 - cond_p)*exp(-norm_weights * x) + cond_p
            if order == 0:
                return np.sum(log(tmp) + norm_weights * x)
            elif order == 1:
                return np.sum(norm_weights * cond_p / tmp)
            elif order == 2:
                return np.sum((1- cond_p)*norm_weights**2 * cond_p * exp(-norm_weights * x) / tmp**2)
            elif order == 3:
                tmp2 = (1- cond_p)*norm_weights**2 * cond_p * exp(-norm_weights * x)
                return np.sum(tmp2 * norm_weights / tmp**2 - 2*tmp2 * norm_weights*cond_p / tmp**3)
            elif order == 4:
                tmp2 = (1- cond_p)*norm_weights**4 * cond_p * exp(-norm_weights * x)
                return np.sum(tmp2/tmp**2 - 6*tmp2*cond_p / tmp**3 + 6*tmp2 * cond_p**2 / tmp**4)
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

    def tail_expectation(self, x): # E[(X-K)1_{X>K}]
        gma = MyGamma(self.shape + 1, self.scale)
        res = self.shape*self.scale*(1.0 - gma.cdf(x)) - x*(1.0-self.cdf(x))
        #print('x: {}, res: {}'.format(x, res))
        return res

    def transform(self, x):
        from numpy import exp, log
        mu = self.CGF(0, 1)
        return 1/self.scale - self.shape / exp(x+log(mu)-mu)


class MyInvGauss(MyDistribution):
    def __init__(self, shape, scale): # lambda, mu
        self.shape = shape 
        # scale / shape
        self.scale = scale 
        # shape

    def density(self, x):
        from scipy.stats import invgauss
        return invgauss.pdf(x, self.scale / self.shape, loc=0, scale=self.shape)

    def cdf(self, x):
        from scipy.stats import invgauss
        return invgauss.cdf(x, self.scale / self.shape, loc=0, scale=self.shape)

    def CGF(self, x, order = 0):
        from numpy import log, sqrt
        if order == 0:
            return self.shape/self.scale * (1- sqrt(1.0-2.0*self.scale**2*x/self.shape))
        else:
            f = lambda x, n: x if n <= 0 else f(x, n - 1)*(x-n)
            return -self.shape/self.scale * (-2*self.scale**2/self.shape)**order * (1-2*self.scale**2*x/self.shape)**(0.5 - order) * f(0.5, order - 1)

    def tail_expectation(self, x): # E[X1_{X>K}]
        from scipy.stats import norm
        from numpy import sqrt, exp
        return self.scale*(norm.cdf(-sqrt(self.shape/x)*(x/self.scale - 1)) + exp(2*self.shape/self.scale) * norm.cdf(-sqrt(self.shape/x)*(x/self.scale + 1)))

    def transform(self, x):
        from numpy import exp, log
        mu = self.CGF(0, 1)
        return self.shape/2*(1.0/self.scale**2 - 1.0/exp(2*(x+log(mu)-mu)))
        #return 1/(2*self.scale**2/self.shape)*(1.0 - exp(-x))

class MyGME(MyDistribution):
    def __init__(self, lam):
        self.lam = lam

    def density(self, x):
        from scipy.stats import norm
        from numpy import exp
        return self.lam * exp(self.lam**2/2.0 + self.lam*x - 1) * norm.cdf(-x-self.lam + 1.0/self.lam) + \
            norm.pdf(1/self.lam - x) - exp(self.lam*x - 1 + self.lam**2/2.0) * norm.pdf(1/self.lam - x - self.lam)

    def cdf(self, x):
        from scipy.stats import norm
        from numpy import exp
        return 1.0 - norm.cdf(1/self.lam - x) + exp(self.lam*x - 1 + self.lam**2/2.0)*norm.cdf(1/self.lam - x - self.lam)

    def CGF(self, x, order = 0):
        from numpy import log
        from math import factorial
        if order == 0:
            return x**2/2 + x/self.lam + log(self.lam/(self.lam + x))
        elif order == 1:
            return x + 1/self.lam - 1/(self.lam + x)
        elif order == 2:
            return 1 + 1/(self.lam + x)**2
        else:
            return (-1)**order * factorial(order)*(self.lam + x)**(-order)

    def tail_expectation(self, x): # E[(X-K)1_{X>K}]
        from numpy import exp
        from scipy.stats import norm
        res = norm.pdf(x-1/self.lam)+(1/self.lam-x)*norm.cdf(1/self.lam-self.lam-x)*exp(self.lam*x-1+self.lam**2/2) - x*(1.0 - self.cdf(x))
        #print('x: {}, res: {}'.format(x, res))
        return res

    def transform(self, x):
        from numpy import sqrt
        c = x + self.lam - 1.0/self.lam
        return sqrt(c**2 / 4 + 1) + c / 2 - self.lam

class KouQV(MyDistribution):
    def __init__(self, sigma, lam, etap, etan, prob, rate, div):
        self.sigma = sigma
        self.lam = lam
        self.etap = etap
        self.etan = etan
        self.prob = prob
        self.rate = rate
        self.div = div

    #def CGF(self, x, order = 0):
    #    from numpy import log, exp, inf, real, sqrt
    #    from scipy.integrate import quad
    #    from math import pi
    #    from scipy.stats import norm
    #    sigma = self.sigma
    #    lam = self.lam
    #    etap= self.etap
    #    etan= self.etan
    #    prob= self.prob
    #    rate= self.rate
    #    div= self.div
    #    if (real(x) > 0): return inf
    #    bump = 0.001
    #    if order <= 0: #analytic
    #        c = -x
    #        if real(c) < 0.1: return lam*(2*x*(prob/etap**2+(1-prob)/etan**2))+sigma**2*x #approximation to avoid overflow
    #        return lam*(prob*etap*exp(etap**2/4/c+log(sqrt(pi/c))+log(norm.cdf(-etap/sqrt(2*c)))) + \
    #            (1-prob)*etan*exp(etan**2/4/c+log(sqrt(pi/c))+log(norm.cdf(-etan/sqrt(2*c)))) - 1.0) + sigma**2*x
    #    elif order == 1:
    #        f = lambda z: z**2*exp(x*z**2)
    #        res = sigma**2
    #        y, err = quad(lambda z: f(z)*prob*etap*exp(-etap*z), 0.0, inf)
    #        res += y*lam
    #        y, err = quad(lambda z: f(z)*(1-prob)*etan*exp(etan*z), -inf, 0.0)
    #        res += y*lam
    #        return res
    #    elif order == 2:
    #        dx = max([bump, x*bump])
    #        return (self.CGF(x+dx, order=0) + self.CGF(x-dx, order=0) - 2*self.CGF(x, order=0))/dx**2
    #    elif order == 3:
    #        dx = max([bump, x*bump])
    #        return (self.CGF(x+2*dx, order=0) - 2*self.CGF(x+dx, order=0) + 2*self.CGF(x-dx, order=0) - self.CGF(x-2*dx, order=0))/dx**3/2
    #    elif order == 4:
    #        dx = max([bump, x*bump])
    #        return (self.CGF(x+2*dx, order=0) - 4*self.CGF(x+dx, order=0) + 6*self.CGF(x,order=0) - 4*self.CGF(x-dx, order=0) + self.CGF(x-2*dx, order=0))/dx**4
    #    else:
    #        raise('{}-th derivative not available'.format(order))

    def CGF(self, x, order = 0):
        from numpy import log, exp, inf, real, sqrt
        from scipy.integrate import quad
        from math import pi
        from scipy.stats import norm
        sigma = self.sigma
        lam = self.lam
        etap= self.etap
        etan= self.etan
        prob= self.prob
        rate= self.rate
        div= self.div
        if (real(x) > 0): return inf
        if order == 0:
            f = lambda z: exp(x*z**2)-1
            res = sigma**2 * x
        elif order == 1:
            f = lambda z: z**2*exp(x*z**2)
            res = sigma**2
        elif order == -1: #analytic
            c = -x
            if real(c) < 0.1: return lam*(2*x*(prob/etap**2+(1-prob)/etan**2))+sigma**2*x #approximation to avoid overflow
            return lam*(prob*etap*exp(etap**2/4/c+log(sqrt(pi/c))+log(norm.cdf(-etap/sqrt(2*c)))) + \
                (1-prob)*etan*exp(etan**2/4/c+log(sqrt(pi/c))+log(norm.cdf(-etan/sqrt(2*c)))) - 1.0) + sigma**2*x
        else:
            f = lambda z: z**(2*order)*exp(x*z**2)
            res = 0
        y, err = quad(lambda z: f(z)*prob*etap*exp(-etap*z), 0.0, inf)
        #g = lambda x, z: f(z)*prob*etap*exp(-etap*z)
        #from myfunctions import MyFuncRangeByLeggauss
        #y = MyFuncRangeByLeggauss(0, g, 0, 2048, 30)
        res += y*lam
        y, err = quad(lambda z: f(z)*(1-prob)*etan*exp(etan*z), -inf, 0.0)
        res += y*lam
        return res

class SVJQV(MyDistribution):
    def __init__(self, params):
        self.params = params

    def CGF(self, x, order = 0):

        if order == 0 or order < 0:
            if order < 0:
                from sympy import log, exp, sqrt
            else:
                from numpy import log, exp, sqrt
            theta, kappa, epi, rho, mu, eta, lam, nu, delta, r, x0, v0 = self.params
            phi = 0
            b = 0
            z = x
            gam = 0

            tau = 1

            zeta = sqrt((kappa-phi*rho*epi)**2 + epi**2*(phi-phi**2-2*z))
            psi_ = (kappa-phi*rho*epi)+zeta
            psi = epi**2*(phi-phi**2-2*z)/psi_

            B = (-(phi-phi**2-2*z)*(1-exp(-zeta*tau)) + b*(psi_*exp(-zeta*tau)+psi))/ \
                ((psi+epi**2*b)*exp(-zeta*tau)+psi_-epi**2*b)

            lg = ( (psi+epi**2*b)*exp(-zeta*tau) + psi_-epi**2*b ) / (2*zeta)
            Gamma = r*tau*phi-kappa*theta/epi**2.*(psi*tau + 2*log(lg)) + gam

            k1 = psi + epi**2*b
            k2 = psi_- epi**2*b
            k3 = (1-phi*nu*eta)*k1 - eta*(phi-phi**2-2*z+b*psi_)
            k4 = (1-phi*nu*eta)*k2 - eta*(b*psi-phi+phi**2+2*z)
            factor = exp(z*mu**2./(1-2*delta**2.*z))/sqrt(1-2*delta**2*z)
            if k3 == 0:
                Lambda = - lam*( phi*( exp(mu+delta**2/2)/(1-nu*eta)-1 )+1 )*tau+ \
                    lam*factor*( k2/k4*tau + (k1/k4)* (1-exp(-zeta*tau))/zeta)
            elif k4 == 0:
                Lambda = - lam*( phi*( exp(mu+delta**2/2)/(1-nu*eta)-1 )+1 )*tau+ \
                    lam*factor*( k1/k3*tau + k2/k3*exp(zeta*tau)*(1-exp(-zeta*tau))/zeta )
            else:
                tmp = k1/k3;
                Lambda = - lam*( phi*( exp(mu+delta**2/2)/(1-nu*eta)-1 )+1 )*tau+ \
                    lam*factor*( k2*tau/k4 - (tmp-k2/k4)/zeta*log( (k3*exp(-zeta*tau)+k4)/(k3+k4) ))
            return phi*x0 + B*v0 + z*0 + Gamma + Lambda

        else:
            dx = max(0.0055, 0.001*abs(x))
            if order == 1:
                #from sympy.utilities.lambdify import lambdify
                #from sympy import symbols, diff
                #s = symbols('s')
                #fs = self.CGF(s, order = -1)
                #fs = diff(fs, s)
                #tmp = fs.evalf(subs={s: x})
            
                res = (self.CGF(x+dx, order=0) - self.CGF(x-dx, order=0))/2/dx
                #print(tmp, res)
                return res
            elif order == 2:
                return (self.CGF(x+dx, order=0) + self.CGF(x-dx, order=0) - 2*self.CGF(x, order=0))/dx**2
            elif order == 3:          
                return (self.CGF(x+2*dx, order=0) - 2*self.CGF(x+dx, order=0) + 2*self.CGF(x-dx, order=0) - self.CGF(x-2*dx, order=0))/dx**3/2
            elif order == 4:          
                return (self.CGF(x+2*dx, order=0) - 4*self.CGF(x+dx, order=0) + 6*self.CGF(x,order=0) - 4*self.CGF(x-dx, order=0) + self.CGF(x-2*dx, order=0))/dx**4
            else:
                raise('{}-th derivative not available'.format(order))

    def transform(self, x):
        from numpy import sqrt, exp
        from scipy.stats import norm
        theta, kappa, epi, rho, mu, eta, lam, nu, delta, r, x0, v0 = self.params
        upper = min(1.0/2/delta**2, kappa**2/epi**2/2, kappa/eta/2.0)
        p = norm.cdf(x)
        #return x
        return upper*p + (x - upper)*(1.0-p)

#class MyGME2(MyDistribution): # worse performance
#    def __init__(self, lam, alpha = 1.0):
#        self.lam = lam
#        self.alpha = alpha

#    def density(self, x):
#        from scipy.stats import norm
#        from numpy import exp
#        return self.lam * exp(self.alpha**2*self.lam**2/2.0 + self.lam*x - 1) * norm.cdf((1.0/self.lam-x)/self.alpha-self.lam*self.alpha) + \
#            norm.pdf((1/self.lam - x)/self.alpha)/self.alpha - 1.0/self.alpha*exp(self.lam*x - 1 + self.alpha**2*self.lam**2/2.0) * norm.pdf((1.0/self.lam-x)/self.alpha-self.lam*self.alpha)

#    def cdf(self, x):
#        from scipy.stats import norm
#        from numpy import exp
#        return 1.0 - norm.cdf((1/self.lam - x)/self.alpha) + exp(self.lam*x - 1 + self.alpha**2*self.lam**2/2.0)*norm.cdf((1.0/self.lam-x)/self.alpha-self.lam*self.alpha)

#    def CGF(self, x, order = 0):
#        from numpy import log
#        from math import factorial
#        if order == 0:
#            return self.alpha**2*x**2/2 + x/self.lam + log(self.lam/(self.lam + x))
#        elif order == 1:
#            return self.alpha**2*x + 1/self.lam - 1/(self.lam + x)
#        elif order == 2:
#            return self.alpha**2 + 1/(self.lam + x)**2
#        else:
#            return (-1)**order * factorial(order)*(self.lam + x)**(-order)

#    def tail_expectation(self, x): # E[(X-K)1_{X>K}]
#        from numpy import exp
#        from scipy.stats import norm
#        res = norm.pdf(x-1/self.lam)+1/self.lam*norm.cdf(1/self.lam-self.lam-x)*exp(self.lam*x-1+self.lam**2/2) - x*(1.0 - self.cdf(x))
#        #print('x: {}, res: {}'.format(x, res))
#        return res

#    def transform(self, x):
#        from numpy import sqrt
#        c = x + self.alpha**2*self.lam - 1.0/self.lam
#        return (sqrt(c**2 / 4/self.alpha**2 + 1) + c / 2/self.alpha)/self.alpha - self.lam