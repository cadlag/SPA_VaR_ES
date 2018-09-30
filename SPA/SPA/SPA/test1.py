import unittest
import warnings
from SPA import*
from mydistribution import *

class Test_test1(unittest.TestCase):
    def test_Wood(self):
        print("test_Wood")
        gma = MyGamma(1.0, 1.0)
        norm = MyNormal(0.0, 1.0)
        spa_ng = SPANonGaussian_Wood(gma, norm)
        spa = SPA_LR(gma)

        K = 1.
        p1 = spa.approximate(K)
        p2 = spa_ng.approximate(K)
        print(p1, p2)
        self.assertTrue(abs(p1 - p2) < 1e-4)

        K = 1.2
        p1 = spa.approximate(K)
        p2 = spa_ng.approximate(K)
        print(p1, p2)
        self.assertTrue(abs(p1 - p2) < 1e-4)

        K = 0.7
        p1 = spa.approximate(K)
        p2 = spa_ng.approximate(K)
        print(p1, p2)
        self.assertTrue(abs(p1 - p2) < 1e-4)

    def test_ZK(self):
        gma = MyGamma(1.0, 1.0)
        norm = MyNormal(0.0, 1.0)
        spa_ng = SPANonGaussian_ZK(gma, norm)
        spa2 = SPA_ButlerWood(gma)
        spa3 = SPA_Martin(gma)
        spa4 = SPA_Studer(gma)

        print("test_ZK1")
        K = [0.9, 1.0, 1.1]
        for k in K:
            print(gma.tail_expectation(k), spa_ng.approximate(k), spa2.approximate(k,2), spa3.approximate(k, 2))
        #self.assertTrue(abs(p1 - p2) < 1e-4)

    def test_ZK2(self):
        gma = MyGamma(1.0, 1.0)
        gma2 = MyGamma(1.0, 2.0)
        spa_ng = SPANonGaussian_ZK(gma2, gma)

        K = [0.9, 1.0, 1.1]
        print("test_ZK2")
        for k in K:
            print(gma2.tail_expectation(k), spa_ng.approximate(k))

    def test_ZK3(self):
        gma = MyGamma(1.0, 1.0)
        norm = MyNormal(0.0, 1.0)
        spa_ng = SPANonGaussian_HO(gma, norm)
        spa1 = SPA_ButlerWood(gma)

        print("test_ZK3")
        K = [0.9, 1.0, 1.1]
        for k in K:
            print(gma.tail_expectation(k), spa_ng.approximate(k), spa1.approximate(k), spa1.approximate(k, 2))

    def test_GetBaseDist(self):
        print("test_GetBaseDist")
        gma = MyGamma(1.0, 1.0)
        spa_ng = SPANonGaussian(gma)
        self.assertTrue(isinstance(spa_ng.getBaseDist(1.0), MyNormal))

        spa_ng = SPANonGaussian(gma, "gamma")
        self.assertTrue(isinstance(spa_ng.getBaseDist(1.0), MyGamma))
        K = 1.0
        z_h = spa_ng.getSaddlepoint(K)
        w_h = spa_ng.getSaddlepoint2(K)
        base = spa_ng.getBaseDist(K)
        xi_b_4 = base.CGF(w_h, 4) / base.CGF(w_h, 2)**2
        xi_d_4 = gma.CGF(z_h, 4) / gma.CGF(z_h, 2)**2
        print("gamma", xi_b_4, xi_d_4)
        self.assertTrue(xi_b_4 == xi_d_4, "cumulant4 does not match")

        spa_ng = SPANonGaussian(gma, "invgauss")
        z_h = spa_ng.getSaddlepoint(K)
        w_h = spa_ng.getSaddlepoint2(K)
        base = spa_ng.getBaseDist(K)
        xi_b_4 = base.CGF(w_h, 4) / base.CGF(w_h, 2)**2
        xi_d_4 = gma.CGF(z_h, 4) / gma.CGF(z_h, 2)**2
        print("invgauss", xi_b_4, xi_d_4)
        self.assertTrue(abs(xi_b_4 - xi_d_4) <= 1e-6, "cumulant4 does not match")

    def test_InvGauss(self):
        print("test_ingauss")
        from numpy import sqrt, exp
        from math import pi
        lam = 5.0
        mu = 1.0
        f = lambda x: sqrt(lam / (2*pi*x**3)) * exp(-lam*(x-mu)**2/(2*mu**2*x))
        invg = MyInvGauss(lam, mu)
        f1 = invg.density(2.5)
        f2 = f(2.5)
        print(f1, f2)
        self.assertTrue(abs(f1 - f2) <= 1e-6)

        v1 = (invg.CGF(1.001, 0) - invg.CGF(0.999, 0))/0.002
        v2 =  invg.CGF(1.0, 1)
        print("CGF", v1, v2)
        self.assertTrue(abs(v1 - v2) < 1e-6)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        unittest.main(exit = False)
