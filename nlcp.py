__doc__ = 'nonlinear complementarity problem solver on NMF problem'

import numpy


def fixed_point(M, p, tol=1e-4, maxiter=1000):
    '''
    Given a m*n matrix M, compute a non negative
    matrix factorization by minimizing Frobenius
    norm, i.e.:
    
    min || M - V * H ||_{F}
    
    where:  V is m*p and V >= 0,
            H is p*n and H >= 0

    This code solve this problem by solving the KKT system:

    V >= 0, H >= 0,
    (VH - M) * H.T >= 0,
    V.T(VH - M) >= 0,
    V .* (VH - M) * H.T = 0,
    H .* V.T(VH - M) = 0

    This use NLCP method, solve
    f(|x| - x) - |x| - x = 0
    '''
    
    tol_cur = 1e8
    m, n = M.shape
    V = numpy.random.randn(m, p)
    H = numpy.random.randn(p, n)
    it = 0
    T = numpy.ones(M.shape)
    status = 'Not Coverage'

    while tol_cur > tol:
        import pdb
        pdb.set_trace()
        absV = numpy.abs(V)
        absH = numpy.abs(H)
        V1 = absV - V
        H1 = absH - H
        tol_cur = numpy.linalg.norm(T - numpy.dot(V, H)) / numpy.linalg.norm(T)
        T = numpy.dot(V1, H1)
        V = numpy.dot(T - M, H1.T) - absV
        H = numpy.dot(V1.T, T - M) - absH
        it += 1
        if it > maxiter:
            break
    if not tol_cur > tol:
        status = 'Converge'
    return {'status':status, 'V':V, 'H': H}



def foo(M, p, tol=1e-4, maxiter=1000):
    m, n = M.shape
    assert m == n, 'only suppoer m == n'
    V = numpy.random.randn(m, p)
    H = numpy.random.randn(p, n)
    it = 0
    T = numpy.dot(V, H)
    status = 'Noe Converge'
    l = m * p
    J_fv = numpy.zeros([l, l])
    J_fh = numpy.zeros([l, l])
    J_gv = numpy.zeros([l, l])
    J_gh = numpy.zeros([l, l])

    def dfdv():
        HHT = numpy.dot(H, H.T)
        for fi in range(m):
            for fk in range(p):
                for vi in range(m):
                    for vk in range(p):
                        J_fv[fi*p+fk, vi*p+vk] = HHT[fk, vk] if vi == fi else 0

    def dfdh():
        for fi in range(m):
            for fk in range(p):
                for hi in range(p):
                    for hk in range(m):
                        J_fh[fi*p+fk, hi*m+hk] = numpy.sum(V[fi, :] * H[:, hk]) + V[fi, hk] * H[hi, hk] + M[fi, hk] if hi == fk else V[fi, hi] * H[fk, hk]

    def dgdv():
        for gi in range(p):
            for gk in range(m):
                for vi in range(m):
                    for vk in range(p):
                        J_gv[gi*m+gk, vi*p+vk] = numpy.sum(V[vi, :] * H[:, gk]) + V[vi, vk] * H[vk, gk] + M[vi, gk] if vk == gi else V[vi, gi] * H[vk, gk]
        return None

    def dgdh():
        VTV = numpy.dot(V.T, V)
        for gi in range(p):
            for gk in range(m):
                for hi in range(p):
                    for hk in range(m):
                        J_gh[gi*m+gk, hi*m+hk] = VTV[gi, hi] if hk == gk else 0
        return None
    return None
