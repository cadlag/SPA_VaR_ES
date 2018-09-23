from numpy.polynomial.legendre import leggauss

def MyFuncByLeggauss(x, func, bd=10, deg=30):

    print('input x = {}'.format(x))

    samples, weights = leggauss(deg)

    res = 0.0

    for i in range(len(samples)):
        tmp = func(x, bd * samples[i]) * bd
        res +=  tmp * weights[i]
    print('output y = {}'.format(res))

    return res
