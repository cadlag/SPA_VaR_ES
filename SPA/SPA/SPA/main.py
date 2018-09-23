# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 15:38:28 2017

@author: Daniel
"""

from SPA import *
from myfunctions import *
from mydistribution import *
from vasicek import *
import numpy as np
from math import sqrt, log, exp
from scipy.optimize import fsolve, brentq, newton
from scipy import integrate
from scipy.stats import norm
from time import time
from numpy.polynomial.legendre import leggauss
import pandas as pd

nonGaussian = True

if nonGaussian == False:
    my_norm = MyNormal(1, 3)
    LR = SPA_LR(my_norm)

    #for x in np.linspace(-10, 10, num = 50):
    #    print LR.approximate(x), 1 - my_norm.cdf(x)

    avg_loss = ConditionalLossDist(np.ones(100), 0.01*np.ones(100), 0.0*np.ones(100))

    print (0.5, avg_loss.CGF(0, 1))
    print (250 / 1e6, avg_loss.CGF(0, 2))

    func1 = lambda x: SPA_LR(avg_loss).approximate(x) - 0.5

    #print func1(0.4), func1(0.5), func1(0.6)
    #print brentq(func1, 0.4, 0.6)

    #start = time()
    #for x in range(10):
    #    print avg_loss.CGF(x, 1)
    #end = time()
    #print 'time elapsed {} for for loop'.format(end - start)

    #start = time()
    #for x in range(10):
    #    print avg_loss.CGF_ary(x, 1)
    #end = time()
    #print 'time elapsed {} for array ops'.format(end - start)

    ##print SPA_LR(avg_loss).approximate(0.5, discrete = True)
    #print SPA_LR(avg_loss).approximate(0.5, discrete = False)

    func_inner = lambda x, y: SPA_LR(avg_loss.setY(y)).approximate(x)
    #func_outer = lambda y: func_inner(0.5,y) * norm.pdf(y)

    func = lambda x,y: func_inner(x, y)* norm.pdf(y)

    #start = time()
    #target_func = lambda x: MyFuncByLeggauss(x, func) - 0.05

    #print 'root finding starts...'

    ##res = brentq(target_func, 0.3, 0.7)
    ##res = fsolve(target_func, 0.5)
    #res = newton(target_func, 0.5)
    #print 'root found: {}.'.format(res)
    #end = time()
    #print end - start

    print('testing SP formulas for fixed y = 0...')

    studer = StuderTiltedDist(avg_loss)
    spa_studer = SPA_Studer(avg_loss)
    spa_martin = SPA_Martin(avg_loss)
    spa_huang = SPA_ButlerWood(avg_loss)

    #x = 0.015
    ##print avg_loss.CGF(0, 1)
    #print spa_studer.approximate(x)
    #print spa_martin.approximate(x), spa_martin.approximate(x, 2)
    #print spa_huang.approximate(x), spa_huang.approximate(x, 2)

    #print 'testing ES formulas fixed y integration...'

    #func_inner = lambda x, y: SPA_Studer(avg_loss.setY(y)).approximate(x)
    #func = lambda x,y: func_inner(x, y)* norm.pdf(y)

    #alpha = 0.05
    #var = 0.58

    #start = time()
    #es_studer = MyFuncByLeggauss(var, func) / alpha
    #end = time()

    #print es_studer, end - start

    print('testing vasicek...')

    n = 100
    weights = np.empty(n)
    weights[:0.2*n] = 1
    weights[0.2*n:0.4*n] = 4
    weights[0.4*n:0.6*n] = 9
    weights[0.6*n:0.8*n] = 16
    weights[0.8*n:] = 25

    #weights = np.ones(n)

    a = sum(weights)

    vasicek = VasicekOneFactor(weights, 0.01*np.ones(n), 0.2*np.ones(n))

    #start = time()
    #print 'vasicek VaR: {} by closed form.'.format(vasicek.calcVaRFormula(alpha = 0.001))
    #print 'vasicek VaR MC: {}'.format(vasicek.calcVaRMC(alpha = 0.001)[0]*a)
    #print 'vasicek VaR {}...'.format(vasicek.calcVaR(alpha = 0.001)*a)
    #end = time()

    #print end - start

    alphas = np.array([0.001, 0.01, 0.05, 0.1, 0.25])
    #alphas = alphas[0:1]

    VaR_MC = np.empty(alphas.size)
    VaR_SP = np.empty(alphas.size)
    ES_MC = np.empty(alphas.size)
    ES_SP_Studer = np.empty(alphas.size)
    ES_SP_Martin1 = np.empty(alphas.size)
    ES_SP_BW = np.empty(alphas.size)
    ES_SP_KK = np.empty(alphas.size)

    for i in range(alphas.size):
        start = time()

        VaR_MC[i], ES_MC[i] = vasicek.calcVaRMC(alpha = alphas[i], loops = 100000)
        VaR_MC[i] *= a
        ES_MC[i] *= a

        #VaR_SP[i] = vasicek.calcVaR(alpha = alphas[i])*a

        #ES_SP_Studer[i] = vasicek.calcES('spa_studer', alpha = alphas[i])*a
        #ES_SP_Martin1[i] = vasicek.calcES('spa_martin', alpha = alphas[i])*a
        #ES_SP_KK[i] = vasicek.calcES('spa_martin', order = 2, alpha = alphas[i])*a
        #ES_SP_BW[i] = vasicek.calcES('spa_butlerwood', order = 2, alpha = alphas[i])*a

        end = time()

        print(i, end - start)
        print('VaR_MC: {}, VaR_SP: {}'.format(VaR_MC[i], VaR_SP[i]))
        print('ES_MC: {}'.format(ES_MC[i]))
        print('ES_Studer: {}'.format(ES_SP_Studer[i]))
        print('ES_Martin: {}'.format(ES_SP_Martin1[i]))
        print('ES_KK: {}'.format(ES_SP_KK[i]))
        print('ES_BW: {}'.format(ES_SP_BW[i]))

    df = pd.DataFrame(np.array([VaR_MC, VaR_SP, ES_MC, ES_SP_Studer, ES_SP_Martin1, ES_SP_KK, ES_SP_BW]).T, \
        index = 1 - alphas, columns = ['VaR_MC', 'VaR_SP', 'ES_MC', 'ES_Studer', 'ES_Martin', 'ES_KK', 'ES_BW'])
    df.index.name = 'Alpha'

    df.to_csv('VaR_ES.csv')

else:
    gma = MyGamma(1,0.5)
    norm = MyNormal(1, 1)
    norm0 = MyNormal(0,1)
    spa_ng = SPANonGaussian(norm, norm0)
    print(spa_ng.getSaddlepoint(0.5))
    print(spa_ng.getSaddlepoint2(0.5))
    print(spa_ng.getSaddlepoint(1.5))
    print(spa_ng.getSaddlepoint2(1.5))