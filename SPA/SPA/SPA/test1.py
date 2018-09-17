import unittest
from SPA import*
from mydistribution import *

class Test_test1(unittest.TestCase):
    def test_Wood(self):
        gma = MyGamma(1.0, 1.0)
        norm = MyNormal(0.0, 1.0)
        spa_ng = SPANonGaussian_Wood(gma, norm)
        spa = SPA_LR(gma)
        p1 = spa.approximate(1.0)
        print(p1)
        p2 = spa_ng.approximate(1.0)
        print(p2)
        self.assertTrue(abs(p1 - p2) < 1e-4)

if __name__ == '__main__':
    unittest.main()
