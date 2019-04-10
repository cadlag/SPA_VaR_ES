
from SPA import *
from myfunctions import *
from mydistribution import *
from scipy import integrate
from numpy import sign
from scipy.stats import norm
from scipy.optimize import brentq, newton

import numpy as np

class VasicekOneFactor(object):

    def __init__(self, weights, probs, corrs, y_dist= MyNormal()):
        self.weights_ = weights
        self.probs_ = probs
        self.corrs_ = corrs
        self.y_dist_ = y_dist

    def sf(self, x, baseDist = None):
        print('calculating P(L>x)...')

        cond_loss = ConditionalLossDist(self.weights_,self.probs_,self.corrs_)
        if baseDist == None:
            func_inner = lambda x,y: SPA_LR(cond_loss.setY(y)).approximate(x) * self.y_dist_.density(y)
        else:
            func_inner = lambda x,y: SPANonGaussian_Wood(cond_loss.setY(y), baseDist).approximate(x) * self.y_dist_.density(y)
        target_func = lambda x: MyFuncByLeggauss(x, func_inner, bd = 4, deg = 50)

        return target_func(x)

    def calcVaR(self, alpha = 0.05, baseDist = None):

        assert(alpha > 0 and alpha < 1)
        print('calculating VaR...')

        cond_loss = ConditionalLossDist(self.weights_,self.probs_,self.corrs_)
        if baseDist == None:
            func_inner = lambda x,y: SPA_LR(cond_loss.setY(y)).approximate(x) * self.y_dist_.density(y)
        else:
            func_inner = lambda x,y: SPANonGaussian_Wood(cond_loss.setY(y), baseDist).approximate(x) * self.y_dist_.density(y)
        target_func = lambda x: MyFuncByLeggauss(x, func_inner, bd = 4, deg = 50) - alpha           

        guess = cond_loss.CGF(0, 1)
        sgn = sign(target_func(guess))          
        
        if sgn == 0:
            res = guess
        else:
            a = 1.0 / guess - 1
            i = 1
            while sign(target_func(1.0 / (1.0 + a * 2 ** (-sgn * i)))) == sgn:
                i += 1
            res = brentq(target_func, 1.0 / (1.0 + a * 2 ** (-sgn * (i - 1))), 1.0 / (1.0 + a * 2 ** (-sgn * i)), disp=True)
        #self.var_ = res
        return res

    def calcES(self, spa_type, order = 1, alpha = 0.05, baseDist = None):

        print('calculating ES using {}...'.format(spa_type))
        #try:
        #    K = self.var_
        #except:
        #    self.calcVaR(alpha = alpha, baseDist=baseDist)
        #    K = self.var_
        K = self.calcVaR(alpha = alpha, baseDist=baseDist)
        cond_loss = ConditionalLossDist(self.weights_, self.probs_, self.corrs_)

        if spa_type.lower() == 'spa_martin':           
            func_inner = lambda x, y: SPA_Martin(cond_loss.setY(y)).approximate(x) * self.y_dist_.density(y)
        elif spa_type.lower() == 'spa_studer':
            func_inner = lambda x, y: SPA_Studer(cond_loss.setY(y)).approximate(x) * self.y_dist_.density(y)
        elif spa_type.lower() == 'spa_butlerwood':
            func_inner = lambda x, y: SPA_ButlerWood(cond_loss.setY(y)).approximate(x) * self.y_dist_.density(y)
        elif spa_type.lower() == 'spanongaussian_zk':
            func_inner = lambda x, y: SPANonGaussian_ZK(cond_loss.setY(y), baseDist).approximate(x) * self.y_dist_.density(y)
        elif spa_type.lower() == 'spanongaussian_ho':
            func_inner = lambda x, y: SPANonGaussian_HO(cond_loss.setY(y), baseDist).approximate(x) * self.y_dist_.density(y)
        else:
            raise Exception("spa type " + spa_type + " not supported.")

        return MyFuncByLeggauss(K, func_inner, bd = 4.5) / alpha

    def calcVaRMC(self, alpha = 0.05, loops = 10000):

        assert(alpha > 0 and alpha < 1)
        print('calculating VaR using MC...')  

        # prepare thresholds
        thres = self.y_dist_.ppf(self.probs_)
        weights = self.weights_ / np.sum(self.weights_)       

        s = 0.0
        s2 = 0.0
        ES = 0.0
        ES2 = 0.0
        nsample = 10000

        for i in range(loops):
            rvs = self.y_dist_.rvs(size = nsample * (1 + weights.size))
            y = np.tile(rvs[:nsample], (weights.size, 1)).T
            z = rvs[nsample:].reshape(nsample, -1)
            x = np.sqrt(self.corrs_) * y + np.sqrt(1 - self.corrs_) * z
            loss = np.sum((x < thres)*weights, axis = 1)         
            loss.sort()
            idx = (int)(nsample*(1-alpha))
            tmp = loss[idx - 1]
            tmp1 = loss[idx - 1:].sum() / (nsample*alpha) #why +1?
            s += tmp
            ES += tmp1
            s2 += tmp**2
            ES2 += tmp1**2

            if (i+1) % 100 == 0:
                print('{}: running VaR: {} ({}); ES: {} ({})'.format(i, s / (i + 1), np.sqrt((s2 - s**2/(i+1))/max(1, i)/(i+1)), \
                    ES / (i+1), np.sqrt((ES2 - ES**2 / (i+1))/max(1, i)/(i+1)) ))

        return (s / loops, np.sqrt((s2 - s**2/loops) /(loops - 1) / loops), ES / loops, np.sqrt((ES2 - ES**2 / loops) / (loops-1) / loops))

    def calcVaRFormula(self, alpha = 0.05):
        return np.sum(self.y_dist_.cdf( (self.y_dist_.ppf(self.probs_) + np.sqrt(self.corrs_) * self.y_dist_.ppf(1 - alpha)) \
            / np.sqrt(1 - self.corrs_)) * self.weights_)





