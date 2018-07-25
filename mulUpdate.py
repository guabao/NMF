__doc__ = 'multiplicative update algorithm'

import numpy


def mulUpdateFrobenius(M, p, tol=1e-4, maxiter=1000):
    '''
    given a m*n matrix M, compute a non negative
    matrix factorization by minimizing Frobenius
    norm, i.e.:
    
    min || M - V * H ||_{F}
    
    where:  V is m*p and V >= 0,
            H is p*n and H >= 0

    this code follows the paper:
    http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf

    the update rule is:
    H_{x, y} = H_{x, y} * (V^{T} * M)_{x, y} / (V^{T} * V * H)_{x, y}
    V_{x, y} = V_{x, y} * (M * H^{T})_{x, y} / (V * H * H^{T})_{x, y}
    '''

    tol_cur = 1e8
    m, n = M.shape
    V = numpy.random.rand(m, p)
    H = numpy.random.rand(p, n)
    it = 0
    status = 'Not Converge'
    
    while tol_cur > tol:
        T = numpy.dot(V, H)
        H = H * (numpy.dot(V.T, M)) / (numpy.dot(numpy.dot(V.T, V), H))
        V = V * (numpy.dot(M, H.T)) / (numpy.dot(numpy.dot(V, H), H.T))
        it += 1
        if it > maxiter:
            break
        tol_cur = numpy.linalg.norm(T - numpy.dot(V, H)) / numpy.linalg.norm(T)
    if not tol_cur > tol:
        status = 'Converge'
    return {'status':status, 'V':V, 'H':H}


def mulUpdateKLD(M, p, tol=1e-4, maxiter=1000):
    '''
    given a m*n matrix M, compute a non negative
    matrix factorization by minimizing Kullback
    Leibler Divergence, i.e.:
    
    min     KLD(M || V * H)
    
    where:  V is m*p and V >= 0,
            H is p*n and H >= 0

    this code follows the paper:
    http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf

    the update rule is:
    H_{x, y} = H_{x, y} * (V^{T} * M)_{x, y} / (V^{T} * V * H)_{x, y}
    V_{x, y} = V_{x, y} * (M * H^{T})_{x, y} / (V * H * H^{T})_{x, y}
    '''

    tol_cur = 1e8
    m, n = M.shape
    V = numpy.random.rand(m, p)
    H = numpy.random.rand(p, n)
    it = 0
    status = 'Not Converge'
    
    while tol_cur > tol:
        T = numpy.dot(V, H)
        H = H * numpy.dot((V / numpy.sum(V, aixs=0)).T, V / T)
        V = V * numpy.dot(V / T, H.T) / numpy.sum(H, axis=0)
        it += 1
        if it > maxiter:
            break
        tol_cur = numpy.linalg.norm(T - numpy.dot(V, H)) / numpy.linalg.norm(T)
    if not tol_cur > tol:
        status = 'Converge'
    return {'status':status, 'V':V, 'H': H}



def testMx():
    n = 10
    p = 2
    mx = numpy.round(numpy.random.rand(n, n), 2)
    d = mulUpdateFrobenius(mx, p)
    return mx, d
