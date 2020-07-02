from sympy import Matrix, gcdex
from . import integer_manipulations as int_man
import numpy as np;

def reduce_po_lat(l_dsc_p, l_p_po, tol1=1e-6):
    N0, _ = int_man.int_mult(l_dsc_p);
    l_po_p = l_p_po.inv();
    l_dsc_po = l_p_po*l_dsc_p;

    N1, imat1 = int_man.int_mult(l_dsc_po, tol1);
    ### Check that imat1 is indeed an integer matrix
    imat1 = np.array(imat1, dtype='double')
    cond1 = int_man.check_int_mat(imat1, tol1);
    ### Check that l_dsc_po == imat1/N1
    cond2 = (np.max(np.abs(l_dsc_po - imat1/N1)) < tol1);
    if cond1 and cond2:
        imat2 = (np.around(imat1)).astype(int)
        # imat2 = Matrix(imat1)
        lll_imat2 = lll_reduction(imat2);
        cond3 = check_basis_equi(imat2, lll_imat2);
        if cond3:
            lll_dsc_po = lll_imat2/N1;
            cond4 = check_basis_equi(l_dsc_po, lll_dsc_po);
            if cond4:
                lll_dsc_p = l_po_p*lll_dsc_po;
                if (N0 == 1):
                    N1, lll_dsc_p = int_man.int_mult(lll_dsc_p);
                    if (N1 != 1):
                        raise Exception("Matrix in reference frame 'p' is not an integer matrix.")
                    else:
                        return lll_dsc_p;
                else:
                    return lll_dsc_p;
            else:
                raise Exception("Lattice Reduction did not work.")
        else:
            raise Exception("Lattice Reduction did not work.")
    else:
        raise Exception("Lattice Reduction did not work.")
# -----------------------------------------------------------------------------------------------------------

def check_basis_equi(l_p1, l_p2, tol1 = 1e-10):
    if type(l_p2) == np.ndarray:
        l_p2 = Matrix(l_p2);
    if type(l_p1) == np.ndarray:
        l_p1 = Matrix(l_p1);

    sz = l_p1.shape;
    if sz[0] == sz[1]:
        int_mat = (l_p1.inv())*l_p2;
        cond1 = int_man.check_int_mat(int_mat, tol1);
        tol2 = tol1*np.max(np.abs(int_mat));
        cond2 = ( abs(abs(int_mat.det()) - 1) < tol2);
        if (cond1 and cond2):
            int_mat = (l_p2.inv())*l_p1;
            cond1 = int_man.check_int_mat(int_mat, tol1);
            tol2 = tol1*np.max(np.abs(int_mat));
            cond2 = ( abs(abs(int_mat.det()) - 1) < tol2);
            return (cond1 and cond2);
        else:
            return False;
    else:
        l_p1_inv = (((l_p1.T*l_p1).inv())*l_p1.T)
        int_mat = (l_p1_inv)*l_p2;
        cond1 = int_man.check_int_mat(int_mat, tol1);
        tol2 = tol1*np.max(np.abs(int_mat));
        cond2 = ( abs(abs(int_mat.det()) - 1) < tol2);
        if (cond1 and cond2):
            l_p2_inv = (((l_p2.T*l_p1).inv())*l_p2.T)
            int_mat = (l_p2_inv)*l_p1;
            cond1 = int_man.check_int_mat(int_mat, tol1);
            tol2 = tol1*np.max(np.abs(int_mat));
            cond2 = ( abs(abs(int_mat.det()) - 1) < tol2);
            return (cond1 and cond2);
        else:
            return False;
# -----------------------------------------------------------------------------------------------------------

def check_basis_def(l_p1, l_p2, tol1 = 1e-10):
    """
    l_p2 is defined in l_p1
    (l_p1.inv())*l_p2 is an integer matrix
    """
    if type(l_p2) == np.ndarray:
        l_p2 = Matrix(l_p2);
    if type(l_p1) == np.ndarray:
        l_p1 = Matrix(l_p1);

    sz = l_p1.shape;
    if sz[0] == sz[1]:
        int_mat = (l_p1.inv())*l_p2;
        cond1 = int_man.check_int_mat(int_mat, tol1);
        return(cond1)
    else:
        l_p1_inv = (((l_p1.T*l_p1).inv())*l_p1.T)
        int_mat = (l_p1_inv)*l_p2;
        cond1 = int_man.check_int_mat(int_mat, tol1);
        return (cond1);

# -----------------------------------------------------------------------------------------------------------

def lll_reduction(matrix, delta=0.75):
    """
    This function has been borrowed from pymatgen project with slight changes
    in input and output handling.

    This function has been borrowed from pymatgen project with slight changes
    in input and output handling.

    Link to the project: http://pymatgen.org/

    Link to the source code: https://github.com/materialsproject/pymatgen/blob/92ee88ab6a6ec6e27b717150931e6d484d37a4e6/pymatgen/core/lattice.py

    Performs a Lenstra-Lenstra-Lovasz lattice basis reduction to obtain a
    c-reduced basis. This method returns a basis which is as "good" as
    possible, with "good" defined by orthongonality of the lattice vectors.

    Parameters
    ----------
    delta: float
        Reduction parameter. \v
        Default of 0.75 is usually fine.

    Returns
    -------
    a: numpy array
        Reduced lattice:
    """
    #leila: https://mnsgrg.com/2018/08/09/lattice-basis-reduction-part-1/
    #finding more orthogonal and smaller lattice vector to fill up the space with lattice points

    if int_man.check_int_mat(matrix, 1e-10):

        # Transpose the lattice matrix first so that basis vectors are columns.
        # Makes life easier.
        a = np.array(matrix)
        sz = a.shape

        b = np.zeros((3, 3))  # Vectors after the Gram-Schmidt process
        u = np.zeros((3, 3))  # Gram-Schmidt coeffieicnts
        m = np.zeros(3)  # These are the norm squared of each vec.

        b[:, 0] = a[:, 0]
        m[0] = np.dot(b[:, 0], b[:, 0])
        for i in range(1, sz[1]):
            u[i, 0:i] = np.dot(a[:, i].T, b[:, 0:i]) / m[0:i]
            b[:, i] = a[:, i] - np.dot(b[:, 0:i], u[i, 0:i].T)
            m[i] = np.dot(b[:, i], b[:, i])

        k = 2

        while k <= sz[1]:
            # Size reduction.
            for i in range(k - 1, 0, -1):
                q = round(u[k - 1, i - 1])
                if q != 0:
                    # Reduce the k-th basis vector.
                    a[:, k - 1] = a[:, k - 1] - q * a[:, i - 1]
                    uu = list(u[i - 1, 0:(i - 1)])
                    uu.append(1)
                    # Update the GS coefficients.
                    u[k - 1, 0:i] = u[k - 1, 0:i] - q * np.array(uu)

            # Check the Lovasz condition.
            if np.dot(b[:, k - 1], b[:, k - 1]) >=\
                    (delta - abs(u[k - 1, k - 2]) ** 2) *\
                    np.dot(b[:, (k - 2)], b[:, (k - 2)]):
                # Increment k if the Lovasz condition holds.
                k += 1
            else:
                # If the Lovasz condition fails,
                # swap the k-th and (k-1)-th basis vector
                v = a[:, k - 1].copy()
                a[:, k - 1] = a[:, k - 2].copy()
                a[:, k - 2] = v
                # Update the Gram-Schmidt coefficients
                for s in range(k - 1, k + 1):
                    u[s - 1, 0:(s - 1)] = np.dot(a[:, s - 1].T,
                                            b[:, 0:(s - 1)]) / m[0:(s - 1)]
                    b[:, s - 1] = a[:, s - 1] - np.dot(b[:, 0:(s - 1)],
                                                    u[s - 1, 0:(s - 1)].T)
                    m[s - 1] = np.dot(b[:, s - 1], b[:, s - 1])

                if k > (sz[1]-1):
                    k -= 1
                else:
                    # We have to do p/q, so do lstsq(q.T, p.T).T instead.
                    p = np.dot(a[:, k:sz[1]].T, b[:, (k - (sz[1]-1)):k])
                    q = np.diag(m[(k - (sz[1]-1)):k])
                    result = np.linalg.lstsq(q.T, p.T)[0].T
                    u[k:sz[1], (k - (sz[1]-1)):k] = result

        return a
    else:
        raise Exception(
            'The input to the lll_algorithm is expected to be integral')
# -----------------------------------------------------------------------------------------------------------



#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains factorization algorithms for matrices over the
integers. All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""



# from abelian.utils import mod

def mod(a, b):
    """
    Returns a % b, with a % 0 = a.
    Parameters
    ----------
    a : int
        The first argument in a % b.
    b : int
        The second argument in a % b.
    Returns
    -------
    int
        `a` modulus `b`.
    Examples
    ---------
    >>> mod(5, 2)
    1
    >>> mod(5, 0)
    5
    """
    if b == 0:
        return a
    return a % b

def hermite_normal_form(A):
    """
    Compute U and H such that A*U = H.
    This algorithm computes the column version of the Hermite normal form, and
    returns a tuple of matrices (U, H) such that A*U = H. The matrix U is an
    unimodular transformation matrix and H is the result of the transformation,
    i.e. H is in Hermite normal form.
    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The matrix to decompose.
    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix, i.e. integer matrix with determinant +/- 1.
    H : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A matrix in Hermite normal form.
    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> U, H = hermite_normal_form(A)
    >>> # Verify that U is unimodular (determinant +/- 1)
    >>> U.det() in [1, -1]
    True
    >>> # Verify the decomposition
    >>> A*U == H
    True
    """

    # Get size and set up matrices U and H
    m, n = A.shape
    j = 0
    H = A.copy()
    U = Matrix.eye(n)

    # Iterate down the rows of the matrix
    for i in range(0, m):

        # If every entry to the right is a zero, no pivot will be found
        # for this row and we move on to the next one.
        if H[i, j:] == H[i, j:] * 0:
            continue

        # Create zeros to the right of the pivot H[i, j]
        for k in range(j + 1, n):

            # Skip the column if the element is zero
            # In this case the column index j in not incremented
            if H[i, k] == 0:
                continue

            # Apply the 'elementary hermite transform' to the columns of H,
            # ignoring the top i rows of H as they are identically zero
            # The implementation of the 'elementary hermite transform'
            # does not explicitly compute matrix products.
            # Equivalent to right-multiplication by Matrix([[a, -s/g],
            #                                               [b, r/g]])

            # Extended Euclidean algorithm, so that r*a + s*b = g = gcd(r, s)
            r, s = H[i, j], H[i, k]
            a, b, g = gcdex(r, s)

            # Apply the matrix product of H and U
            H[i:, j], H[i:, k] = (a * H[i:, j] + b * H[i:, k],
                                  -(s / g) * H[i:, j] + (r / g) * H[i:, k])
            U[:, j], U[:, k] = (a * U[:, j] + b * U[:, k],
                                -(s / g) * U[:, j] + (r / g) * U[:, k])

        # Make sure the pivot element is positive.
        # Some saving achieved by realizing that the first i rows of H are
        # identically zero -- thus no multiplication is needed.
        if H[i, j] < 0:
            H[i:, j] = -H[i:, j]
            U[:, j] = -U[:, j]

        # Making all elements to the left of the pivot H[i, j] smaller
        # than the pivot and positive using division algorithm transform
        for k in range(0, j):
            # Compute quotient in the division algorithm, subtracting
            # the quotient times H[:, j] leaves a positive remainder
            a = H[i, k] // H[i, j]
            H[:, k] = H[:, k] - a * H[:, j]
            U[:, k] = U[:, k] - a * U[:, j]
        # Increment j (the column index). Break if j is out of dimension
        j += 1
        if j >= n:
            # j is out of dimension, break
            break

    return U, H

def smith_normal_form(A, compute_unimod = True):
    """
    Compute U,S,V such that U*A*V = S.
    This algorithm computes the Smith normal form of an integer matrix.
    If `compute_unimod` is True, it returns matrices (U, S, V) such
    that U*A*V = S, where U and V are unimodular and S is in Smith normal
    form. If `compute_unimod` is False, it returns S and does not compute U
    and V.
    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The matrix to factor.
    compute_unimod : bool
        Whether or not to compute and return unimodular matrices U and V.
    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix, i.e. integer matrix with determinant +/- 1.
    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A matrix in Smith normal form.
    V : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix, i.e. integer matrix with determinant +/- 1.
    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> U, S, V = smith_normal_form(A)
    >>> # Verify that U and V are both unimodular
    >>> U.det() in [1, -1] and V.det() in [1, -1]
    True
    >>> # Verify the factorization
    >>> U * A * V == S
    True
    >>> # Compute without U and V, verify that the result is the same
    >>> K = smith_normal_form(A, compute_unimod=False)
    >>> K == S
    True
    """

    # Get size and set up the unimodular matrices U and V
    m, n = A.shape
    min_m_n = min(m, n)
    S = A.copy()
    if compute_unimod:
        U, V = Matrix.eye(m), Matrix.eye(n)

    def row_col_all_zero(matrix, f):
        """
        Check that all entries to the right of and below `f` are zero.
        """
        for entry in matrix[f, f + 1:]:
            if entry != 0:
                return False
        for entry in matrix[f + 1:, f]:
            if entry != 0:
                return False
        return True

    # Main loop, iterate over all sub-matrices to reduce
    f = 0
    while f < min_m_n:

        # While there are non-zero elements to reduce in row/column f
        # and the diagonal element is not positive
        while not (row_col_all_zero(S, f) and S[f, f] >= 0):

            # Find index pair of minimum non-zero entry (in absolute value)
            # in the sub-matrix S[f:, f:].
            indices = ((i, j) for j in range(f, n) for i in range(f, m))
            key_val_pairs = ((index, abs(S[index])) for index in indices
                             if abs(S[index]) != 0)
            (i, j), min_val = min(key_val_pairs, key=lambda k: k[1])

            # Permute S to move the minimal element to the pivot location
            S[f:, j], S[f:, f] = S[f:, f], S[f:, j]
            S[i, f:], S[f, f:] = S[f, f:], S[i, f:]
            if compute_unimod:
                V[:, j], V[:, f] = V[:, f], V[:, j]
                U[i, :], U[f, :] = U[f, :], U[i, :]

            # If the freshly permuted pivot is negative, make it positive
            if S[f, f] < 0:
                S[f:, f] = -S[f:, f]
                if compute_unimod:
                    V[:, f] = -V[:, f]

            # Reduce row f so every entry is smaller than pivot
            for k in range(f + 1, n):
                if S[f, k] == 0:
                    continue

                # Subtract a times column f from column k
                a = S[f, k] // S[f, f]
                S[f:, k] = S[f:, k] - a * S[f:, f]
                if compute_unimod:
                    V[:, k] = V[:, k] - a * V[:, f]

            # Reduce column f so every entry is smaller than pivot
            for k in range(f + 1, m):
                if S[k, f] == 0:
                    continue

                # Subtract a times row f from row k
                a = S[k, f] // S[f, f]
                S[k, f:] = S[k, f:] - a * S[f, f:]
                if compute_unimod:
                    U[k, :] = U[k, :] - a * U[f, :]

        f += 1

    # Enforce divisibility criterion using the 'divisibility transformation'
    # matrices.
    for f in range(min_m_n):
        for k in range(f + 1, min_m_n):

            # Divisibility criterion is fulfilled
            if mod(S[k, k], S[f, f]) == 0:
                continue

            # S[f, f] does not divide S[k, k]
            r, s = S[f, f], S[k, k]
            a, b, c = gcdex(r, s)
            S[f, f], S[k, k] = c, (r * s) // c
            if compute_unimod:
                V[:, f], V[:, k] = (V[:, f] + V[:, k],
                                    -b * (s / c) * V[:,f] + a * (r / c) * V[:, k])
                U[f, :], U[k, :] = (a * U[f, :] + b * U[k, :],
                                    -(s / c) * U[f, :] + (r / c) * U[k, :])

    if compute_unimod:
        return U, S, V
    else:
        return S

