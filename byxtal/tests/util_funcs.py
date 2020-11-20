import numpy as np
import byxtal1.integer_manipulations as iman
from sympy.matrices import Matrix, eye, zeros;


def reduce_po_lat(l_csl_p, l_p_po, tol):
    l_p_po = np.array(l_p_po, dtype='double')
    l_csl_po = l_p_po.dot(l_csl_p)
    lInt_csl_po, m1 = iman.int_approx(l_csl_po, tol)
    inp_args = {}
    inp_args['mat'] = lInt_csl_po
    lllInt_csl_po = call_sage_math('./compute_LLL.py', inp_args)
    lllInt_csl_po = Matrix(lllInt_csl_po)
    Sz = lllInt_csl_po.shape
    if Sz[0] == Sz[1]:
        if lllInt_csl_po.det() < 0:
            if Sz[0] == 3:
                M4 = Matrix([[0,1,0],[1,0,0],[0,0,1]])
                lllInt_csl_po = lllInt_csl_po*M4
            if Sz[0] == 2:
                M4 = Matrix([[0,1],[1,0]])
                lllInt_csl_po = lllInt_csl_po*M4

        Tmat = ((Matrix(lInt_csl_po)).inv())*(lllInt_csl_po)
    else:
        A1 = (np.array(lllInt_csl_po, dtype='int64'))
        A2 = (np.array(lInt_csl_po, dtype='int64'))
        # A1inv = np.linalg.pinv(A1)
        A2inv = np.linalg.pinv(A2)
        Tmat = Matrix(np.dot(A2inv, A1))

    cond1 = iman.check_int_mat(Tmat, 1e-12)
    if cond1:
        Tmat1 = Matrix(np.array(np.around(np.array(Tmat, dtype='double')), dtype='int64'))
        return Tmat1
    else:
        raise Exception("Tmat is not an integer matrix.")
