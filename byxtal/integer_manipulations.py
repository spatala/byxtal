import numpy as np

from sympy import Rational
from sympy.matrices import Matrix, eye, zeros;
from sympy import nsimplify
import sympy as spy

import numpy.linalg as nla

def gcd_vec(int_mat):
    input1 = int_mat.flatten()
    Sz = input1.shape
    gcd1 = 0
    for ct1 in range(Sz[0]):
        gcd1 = spy.gcd(gcd1, input1[ct1])

    return int(gcd1)

def gcd_array(input, order='all'):
    """
    The function computes the GCD of an array of numbers.

    Parameters
    ----------
    input : numpy array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)

    order : {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Agcd: numpy array
        An array of greatest common divisors of the input

    Notes
    -------
    * If order = **all**, the input array is flattened and the GCD is calculated
    * If order = **rows**, GCD of elements in each row is calculated
    * If order = **columns** or **cols**, GCD of elements in each column is calculated

    See Also
    --------
    gcd_vec: from fractions module for computing gcd of two integers

    """

    input = np.array(input)
    # Only integer values are allowed
    # if input.dtype.name != 'int64':
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Agcd = gcd_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((sz1,1))
        for ct1 in range(sz1):
            tmp_row = input[ct1,:]
            Agcd[ct1] = gcd_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((1,sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Agcd[0,ct1] = gcd_vec(tmp_row)

    return Agcd

def lcm_vec(Dmat):
    input1 = Dmat.flatten()
    Sz = input1.shape
    lcm1 = 1
    for ct1 in range(Sz[0]):
        lcm1 = spy.lcm(lcm1, input1[ct1])

    return int(lcm1)

def lcm_array(input, order='all'):
    """
    The function computes the LCM of an array of numbers.

    Parameters
    ----------
    input : numpy array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)

    order : {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Alcm: numpy array
        An array of least common multiples of the input

    Notes
    -------
    * If order = **all**, the input array is flattened and the GCD is calculated
    * If order = **rows**, GCD of elements in each row is calculated
    * If order = **columns** or **cols**, GCD of elements in each column is calculated

    See Also
    --------
    lcm_vec: from fractions module for computing gcd of two integers

    """

    input = np.array(input)
    # Only integer values are allowed
    # if input.dtype.name != 'int64':
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Alcm = lcm_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((sz1,1))
        for ct1 in range(sz1):
            tmp_row = input[ct1,:]
            Alcm[ct1] = lcm_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((1,sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Alcm[0,ct1] = lcm_vec(tmp_row)

    return Alcm

def check_int_mat(T, tol1):
    if isinstance(T, Matrix):
        T = np.array(T, dtype='double');
    return (np.max(np.abs(T - np.around(T))) < tol1);

def rat_approx(Tmat, tol1=0.01):
    """
    """
    Tmat = np.array(Tmat)
    input1 = Tmat.flatten()
    nshape = np.shape(Tmat)
    denum_max = 1/tol1
    Sz = input1.shape
    Nmat = np.zeros(np.shape(input1), dtype='int64')
    Dmat = np.zeros(np.shape(input1), dtype='int64')
    for ct1 in range(Sz[0]):
        num1 = (Rational(input1[ct1]).limit_denominator(denum_max))
        Nmat[ct1] = num1.p
        Dmat[ct1] = num1.q

    Nmat1 = np.reshape(Nmat, nshape)
    Dmat1 = np.reshape(Dmat, nshape)

    Nmat1 = np.array(Nmat1, dtype='int64')
    Dmat1 = np.array(Dmat1, dtype='int64')

    return Nmat1, Dmat1;


def int_approx(Tmat, tol1=0.01):
    """
    """
    Tmat = np.array(Tmat)
    tct1 = np.max(np.abs(Tmat))
    tct2 = np.min(np.abs(Tmat))

    mult1 = 1/((tct1 + tct2)/2)
    mult2 = 1/np.max(np.abs(Tmat))

    # print(Tmat)

    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, mult1, tol1)
    int_mat2, t2_mult, err2 = mult_fac_err(Tmat, mult2, tol1)

    if err1 == err2:
        tnorm1 = nla.norm(int_mat1)
        tnorm2 = nla.norm(int_mat2)
        if (tnorm1 > tnorm2):
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult
    else:
        if err1 > err2:
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult

def int_mult_approx(Tmat, tol1=0.01):
    """
    """
    Tmat = np.array(Tmat)
    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, 1, tol1)
    return int_mat1, t1_mult


def mult_fac_err(Tmat, mult1, tol1):
    """
    """
    # print(mult1)
    Tmat1 = Tmat*mult1
    # print(Tmat1)
    N1, D1 = rat_approx(Tmat1, tol1)

    lcm1 = lcm_array(D1)
    N1 = np.array(N1, dtype='double')
    D1 = np.array(D1, dtype='double')

    int_mat1 = np.array((N1/D1)*lcm1, dtype='double')

    cond1 = check_int_mat(int_mat1, tol1*0.01)
    if cond1:
        int_mat1 = np.around(int_mat1)
        int_mat1 = np.array(int_mat1, dtype='int64')
    else:
        raise Exception("int_mat1 is not an integer matrix")
    gcd1 = gcd_vec(int_mat1)
    int_mat1 = int_mat1/gcd1

    int_mat1 = np.array(int_mat1, dtype='int64')
    t1_mult = mult1*lcm1/gcd1
    err1 = np.max(np.abs(Tmat - int_mat1/t1_mult))

    # print(int_mat1)
    # print('+++++++')

    return int_mat1, t1_mult, err1


