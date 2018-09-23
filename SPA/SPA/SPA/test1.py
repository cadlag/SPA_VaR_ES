﻿import unittest
import warnings
from SPA import*
from mydistribution import *

class Test_test1(unittest.TestCase):
    def test_Wood(self):
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


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        unittest.main(exit = False)