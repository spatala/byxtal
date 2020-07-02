# Authors: Arash Dehghan Banadaki <adehgha@ncsu.edu>, Srikanth Patala <spatala@ncsu.edu>
# Copyright (c) 2015,  Arash Dehghan Banadaki and Srikanth Patala.
# License: GNU-GPL Style.
# How to cite GBpy:
# Banadaki, A. D. & Patala, S. "An efficient algorithm for computing the primitive bases of a general lattice plane",
# Journal of Applied Crystallography 48, 585-588 (2015). doi:10.1107/S1600576715004446


import numpy as np
# from fractions import gcd
from math import gcd
from sympy import Rational
from sympy.matrices import Matrix, eye, zeros;
from sympy import nsimplify
# -----------------------------------------------------------------------------------------------------------

def gcd_array(input, order='all'):
    """
    The function computes the GCD of an array of numbers.

    Parameters
    ----------
    input : numpy array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)

    order : {'rows', 'columns', 'col', 'all'}, optional

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
    gcd: from fractions module for computing gcd of two integers

    """

    # Vectorizing the function gcd
    gcd_vec = np.vectorize(gcd)
    tmp = 0

    input = np.array(input)

    if np.ndim(input) == 1:
        input = np.reshape(input, (1, input.shape[0]))

    # Only integer values are allowed
    # if input.dtype.name != 'int64':
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    err_msg = "Not a valid input. Please choose either \"rows\" " + \
              "or \"columns\" keys for this function."

    order_options = ('rows', 'columns', 'col', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Sz = input.shape
        input = np.reshape(input, (1, Sz[0]*Sz[1]))
    if (Keys == 1) or (Keys == 2):
        input = input.T
        Agcd = gcd_array(input, 'rows')
        # handling the case of asking a column
        # vector with the 'row' key by mistake.
        tmp = 1

    Sz = input.shape
    if Sz[1] == 1:
        input.shape = (1, Sz[0])

    Agcd = gcd_vec(input[::, 0], input[::, 1])
    for i in range(Sz[1]-2):
        Agcd = gcd_vec(Agcd, input[::, i+2])

    if tmp != 1:
        Agcd.shape = (len(Agcd), 1)

    return np.absolute(Agcd)
# -----------------------------------------------------------------------------------------------------------


def lcm_vec(a, b):
    """
    The function computes the LCM of two 1D array of integers of length
    and retruns a 1D array of lcm values

    Parameters
    ----------
    a, b : numpy array
        Input 1D arrays of integers

    Returns
    -------
    lcm_vector: numpy array
        Output 1D array of integers

    See Also
    --------
    lcm_arry

    """
    a = a.astype(int); b = b.astype(int);
    gcd_vec = np.vectorize(gcd)
    lcm_vector = a * (b / gcd_vec(a, b))
    return lcm_vector
# -----------------------------------------------------------------------------------------------------------


def lcm_array(input, order='all'):
    """
    The function computes the LCM of an array of numbers.

    Parameters
    ----------
    input : numpy array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)

    order : {'rows', 'columns', 'col', 'all'}, optional

    Returns
    -------
    Alcm: numpy array
        An array of least common multiples of the input

    Notes
    -------
    * If order = **all**, the input array is flattened and the LCM is calculated
    * If order = **rows**, LCM of elements in each row is calculated
    * If order = **columns** or **cols**, LCM of elements in each column is calculated

    See Also
    --------
    gcd_array

    """

    if isinstance(input, Matrix):
        input = np.array(input, dtype='int64');

    input = np.array(input)
    tmp = 0

    if np.ndim(input) == 1:
        input = np.reshape(input, (1, input.shape[0]))

    # Only integer values are allowed
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    err_msg = "Not a valid input. Please choose either \"rows\" " + \
              "or \"columns\" keys for this function."

    order_options = ('rows', 'columns', 'col', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Sz = input.shape
        input = np.reshape(input, (1, Sz[0]*Sz[1]))
    if (Keys == 1) or (Keys == 2):
        input = input.T
        Alcm = lcm_array(input, 'rows')
        # handling the case of asking a column vector
        # with the 'row' key by mistake.
        tmp = 1

    Sz = input.shape
    if Sz[1] == 1:
        input.shape = (1, Sz[0])

    Alcm = lcm_vec(input[::, 0], input[::, 1])
    for i in range(Sz[1]-2):
        Alcm = lcm_vec(Alcm, input[::, i+2])

    if tmp != 1:
        Alcm.shape = (len(Alcm), 1)

    return np.absolute(Alcm)
# -----------------------------------------------------------------------------------------------------------


# def int_check(input, precis=6):
#     """
#     Checks whether the input variable (arrays) is an interger or not.
#     A precision value is specified and the integer check is performed
#     up to that decimal point.

#     Parameters
#     ----------
#     input : numpy array or list
#         Input n-D array of floats.

#     precis : Integer
#         Default = 6.
#         A value that specifies the precision to which the number is an
#         integer. **precis = 6** implies a precision of :math:`10^{-6}`.

#     Returns
#     -------
#     cond: Boolean
#         **True** if the element is an integer to a certain precision,
#         **False** otherwise
#     """

#     var = np.array(input)
#     tval = 10 ** -precis
#     t1 = abs(var)
#     cond = (abs(t1 - np.around(t1)) < tval)
#     return cond
# # -----------------------------------------------------------------------------------------------------------

#############################################################################################################
def int_mult(input, tol=1e-06):
    """
    The function computes the scaling factor required to multiply the
    given input array to obtain an integer array. The integer array is
    returned.

    Parameters
    ----------
    input : numpy array or list of real numbers

    tol : floating point tolerance value
        Default = 1e-06

    Returns
    -------
    N: numpy float array
        An array of integers obtained by scaling input

    Int_Mat: numpy float array
        An array of integers obtained by scaling input

    See Also
    --------
    int_finder

    Notes
    --------
    **Change this function to accept rows and columns as input**
    """

    T = np.array(input, dtype='double');

    n1, d1 = rat_approx(T, tol);
    N = lcm_array(d1); N = N[0][0];
    int_mat = (T*N);
    if check_int_mat(int_mat, tol):
        int_mat = (np.around(int_mat)).astype(int);
        return int(N), Matrix(int_mat);
    else:
        N, int_mat = float_mult(T, tol);
        if check_int_mat(int_mat, tol):
            return N, Matrix(int_mat);
        else:
            raise Exception("Not an integer matrix")

# -----------------------------------------------------------------------------------------------------------
def float_mult(input, tol=1e-6):
    T = np.array(input, dtype='double');

    iarr1 = T.flatten();
    tm1 = np.min(np.abs(iarr1[(np.abs(iarr1) > 1e-12)]));
    tm2 = np.max(np.abs(iarr1));
    m1 = (tm1+tm2)/2;

    T1 = T/m1;
    n1, d1 = rat_approx(T1, tol);
    N = lcm_array(d1); N = N[0][0];
    int_mat = (T1*N);
    if check_int_mat(int_mat, tol):
        int_mat = (np.around(int_mat)).astype(int);
        return int(N)/m1, Matrix(int_mat);
    else:
        ## Add 'from sympy import nsimplify' (use def irrat_approx)
        raise Exception("Not an integer matrix")

# -----------------------------------------------------------------------------------------------------------

def rat_approx(Tmat, tol1):
    """
    """

    input1 = np.array(Tmat)
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    if input1.ndim == 0:
        input1 = np.reshape(input1, (1, 1))

    denum_max = 1/tol1;

    Sz = input1.shape

    Nmat = np.zeros(np.shape(input1), dtype='int64');
    Dmat = np.zeros(np.shape(input1), dtype='int64');

    for ct1 in range(Sz[0]):
        for ct2 in range(Sz[1]):
            num1 = (Rational(input1[ct1,ct2]).limit_denominator(denum_max));
            Nmat[ct1,ct2] = num1.p; Dmat[ct1,ct2] = num1.q;
    return Matrix(Nmat), Matrix(Dmat);
# -----------------------------------------------------------------------------------------------------------

def rat(input, tol1):
    """
    """

    input1 = np.array(input)
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    if input1.ndim == 0:
        input1 = np.reshape(input1, (1, 1))

    denum_max = 1/tol1;

    Sz = input1.shape

    Nmat = np.zeros(np.shape(input1), dtype='int64');
    Dmat = np.zeros(np.shape(input1), dtype='int64');

    for ct1 in range(Sz[0]):
        for ct2 in range(Sz[1]):
            num1 = (Rational(input1[ct1,ct2]).limit_denominator(denum_max));
            Nmat[ct1,ct2] = num1.p; Dmat[ct1,ct2] = num1.q;
    return Nmat, Dmat;
# -----------------------------------------------------------------------------------------------------------

def check_int_mat(T, tol1):
    if isinstance(T, Matrix):
        T = np.array(T, dtype='double');
    return (np.max(np.abs(T - np.around(T))) < tol1);
# -----------------------------------------------------------------------------------------------------------

# def check_equi_basis(l1, l2):
#     if isinstance(l1, np.ndarry):
#         l1 = Matrix(l1);
#     if isinstance(l2, np.ndarry):
#         l2 = Matrix(l2);

#     l1_linv = ((l1.T*l1).inv())*l1.T;
#     cond = check_int_mat((l1_linv*l2), 1e-10)
#     return cond
# # -----------------------------------------------------------------------------------------------------------

def int_finder(Tinp, tol=1e-6):
    """
    """
    if check_int_mat(Tinp, tol):
        Tinp = np.array(Tinp, dtype='double');
        Tinp = np.around(Tinp);
        Tinp = np.array(Tinp, dtype='int64');
        gcd1 = gcd_array(Tinp);
        TI_inp = Tinp/abs(gcd1[0][0]);
    else:
        M1, TI_inp1 = int_mult(Tinp, tol);
        M2, TI_inp2 = float_mult(Tinp, tol);
        if (M2 < M1):
            M = M2; TI_inp = TI_inp2;
        else:
            M = M1; TI_inp = TI_inp1;

    if check_int_mat(TI_inp, tol):
        TI_inp = np.around(np.array(TI_inp, dtype='double'));
        TI_inp = np.array(TI_inp, dtype='int64');
        gcd1 = gcd_array(np.array(TI_inp, dtype='int64'));
        TI_inp = TI_inp/abs(gcd1[0][0]);
        if check_int_mat(TI_inp, tol):
        	TI_inp = np.around(np.array(TI_inp, dtype='double'));
        	TI_inp = np.array(TI_inp, dtype='int64');
        	TI_inp = Matrix(TI_inp);
    else:
        raise Exception("Integer Conversion failed.")

    return TI_inp;

# -----------------------------------------------------------------------------------------------------------

