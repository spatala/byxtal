# Authors: Arash Dehghan Banadaki <adehgha@ncsu.edu>, Srikanth Patala <spatala@ncsu.edu>
# Copyright (c) 2015,  Arash Dehghan Banadaki and Srikanth Patala.
# License: GNU-GPL Style.
# How to cite GBpy:
# Banadaki, A. D. & Patala, S. "An efficient algorithm for computing the primitive bases
# of a general lattice plane",
# Journal of Applied Crystallography 48, 585-588 (2015). doi:10.1107/S1600576715004446

import numpy as np
from . import integer_manipulations as int_man
from . import lll_tools as lt
from sympy.matrices import Matrix, zeros
from sympy import Rational


def find_csl_dsc(l_p_po, T_p1top2_p1, tol1=1e-6, print_check=True):
    """
    This function calls the csl_finder and dsc_finder and returns
    the CSL and DSC basis vectors in 'g1' reference frame.

    Parameters
    -----------------
    L_G1_GO1: numpy array
        The three basis vectors for the primitive unit cell
        (as columns) are given with respect to the GO1 reference
        frame.

    R_G1ToG2_G1: 3X3 numpy array
        The rotation matrix defining the
        transformation in 'G1' reference frame. The subscript 'G1' refers
        to the primitive unit cell of G lattice.

    Returns
    l_csl_g1, l_dsc_g1: numpy arrays
        The basis vectors of csl and dsc lattices in the g1 reference frame
    """
    l_p_po = Matrix(l_p_po)

    T_p1top2_p1 = np.array(T_p1top2_p1, dtype='double')
    Sigma = sigma_calc(T_p1top2_p1, tol1)
    TI_p1top2_p1 = T_p1top2_p1*Sigma
    TI_p1top2_p1 = Matrix((np.around(TI_p1top2_p1)).astype(int))

    l_csl_p = csl_finder(TI_p1top2_p1, Sigma, l_p_po, tol1)
    check_val1 = check_csl(l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)

    l_dsc_p = dsc_finder(T_p1top2_p1, l_p_po, tol1)
    l_p2_p1 = Matrix(T_p1top2_p1)
    print('+++++++++++++++++++++++++++++++++++')
    check_val2 = check_dsc(l_dsc_p, l_csl_p, l_p_po, l_p2_p1, Sigma, print_check)
    print([check_val1, check_val2])
    if (not(check_val1 and check_val2)):
        raise Exception("Error in Computing CSL or DSC Lattices.")
    return l_csl_p, l_dsc_p


def csl_finder(TI_p1top2_p1, Sigma, l_p_po, tol1):
    l_csl_p = csl_finder_smith(TI_p1top2_p1, Sigma)
    l_csl_p = lt.reduce_po_lat(l_csl_p, l_p_po, tol1)
    l_csl_p = make_right_handed(l_csl_p, l_p_po)
    N1, l_csl_p = int_man.int_mult(Matrix(l_csl_p))
    if (N1 != 1):
        raise Exception("CSL Matrix in reference frame 'p' is not an integer matrix.")
    return l_csl_p


def dsc_finder(L_G2_G1, L_G1_GO1, tol1):
    """
    The DSC lattice is computed for the bi-crystal, if the transformation
    matrix l_g2_g1 is given and the basis vectors of the underlying crystal
    l_g_go (in the orthogonal reference go frame) are known. The following
    relationship is used: **The reciprocal of the coincidence site lattice of
    the reciprocal lattices is the DSC lattice**

    Parameters
    ----------------
    l_g2_g1: numpy array
        transformation matrix (r_g1tog2_g1)

    l_g1_go1: numpy array
        basis vectors (as columns) of the underlying lattice expressed in the
        orthogonal 'go' reference frame

    Returns
    ------------
    l_dsc_g1: numpy array
        The dsc lattice basis vectors (as columns) expressed in the g1 reference

    Notes
    ---------
    The "Reduced" refer to the use of LLL algorithm to compute a
    basis that is as close to orthogonal as possible.
    (Refer to http://en.wikipedia.org/wiki/Lattice_reduction) for further
    detials on the concept of Lattice Reduction
    """

    L_GO1_G1 = L_G1_GO1.inv()
    # Reciprocal lattice of G1
    # --------------------------------------------------------------
    L_rG1_GO1 = reciprocal_mat(L_G1_GO1)
    L_GO1_rG1 = L_rG1_GO1.inv()
    # Reciprocal lattice of G2
    # --------------------------------------------------------------
    L_G2_GO1 = L_G1_GO1*L_G2_G1
    L_rG2_GO1 = reciprocal_mat(L_G2_GO1)
    # Transformation of the Reciprocal lattices
    # R_rG1TorG2_rG1 = L_rG2_G1*L_G1_rG1
    L_rG2_rG1 = L_GO1_rG1*L_rG2_GO1
    Sigma_star = sigma_calc(L_rG2_rG1, tol1)
    # Check Sigma_star == Sigma
    LI_rG2_rG1 = L_rG2_rG1*Sigma_star
    if int_man.check_int_mat(LI_rG2_rG1, 1e-10):
        LI_rG2_rG1 = np.around(np.array(LI_rG2_rG1, dtype='double'))
        LI_rG2_rG1 = Matrix(np.array(LI_rG2_rG1, dtype='int64'))
    else:
        raise Exception("Not an integer matrix")
    # CSL of the reciprocal lattices
    L_rCSL_rG1 = csl_finder(LI_rG2_rG1, Sigma_star, L_rG1_GO1, tol1)
    # return L_rG2_rG1, L_rCSL_rG1
    L_rCSL_GO1 = L_rG1_GO1*L_rCSL_rG1
    ####################################################################
    # L_rCSL_rG1 = make_right_handed(L_rCSL_rG1, Matrix(L_rG1_GO1));
    # l1 = Matrix(L_rCSL_rG1);l2 = Matrix(L_rG1_GO1);l3 = Matrix(L_rG2_rG1);
    # check_csl(l1, l2, l3, Sigma_star, True)
    ####################################################################

    L_DSC_GO1 = reciprocal_mat(L_rCSL_GO1)
    L_DSC_G1 = L_GO1_G1*L_DSC_GO1
    if int_man.check_int_mat(L_DSC_G1*Sigma_star, tol1):
        Tmat = np.array(L_DSC_G1*Sigma_star, dtype='double')
        Tmat = np.around(Tmat)
        Tmat = np.array(Tmat, dtype='int64')
        L_DSC_G1 = Matrix(Tmat)/Sigma_star
    else:
        raise Exception("DSC*Sigma is not an integer matrix")

    LLL_DSC_G1 = lt.reduce_po_lat(L_DSC_G1, L_G1_GO1, tol1)
    if int_man.check_int_mat(LLL_DSC_G1*Sigma_star, tol1):
        Tmat = np.array(LLL_DSC_G1*Sigma_star, dtype='double')
        Tmat = np.around(Tmat)
        Tmat = np.array(Tmat, dtype='int64')
        LLL_DSC_G1 = Matrix(Tmat)/Sigma_star
    else:
        raise Exception("DSC*Sigma is not an integer matrix")

    L_DSC_G1 = make_right_handed(LLL_DSC_G1, L_G1_GO1)
    return Matrix(L_DSC_G1)


def csl_finder_smith(rI_g1tog2_g1, Sigma):
    """
    This funciton extracts the CSL basis when transformation between the two
    lattices is given (r_g1tog2_g1). The algorithms used are based on the
    following article: doi:10.1107/S056773947601231X)

    Parameters
    ----------------
    r_g1tog2_g1: numpy array
        The 3x3 transformation matrix in g1 reference frame

    Returns
    -----------
    l_csl_g1: numpy array
        3 x 3 matrix with the csl basis vectors as columns

    Notes
    ---------
    The "Reduced" refer to the use of LLL algorithm to compute a
    basis that is as close to orthogonal as possible.
    (Refer to http://en.wikipedia.org/wiki/Lattice_reduction) for further
    detials on the concept of Lattice Reduction
    """

    A_mat = Matrix(rI_g1tog2_g1)
    # S = U*A_mat*V
    U, S, V = lt.smith_normal_form(A_mat)

    l_p1n_p1 = U.inv()
    T0 = S/Sigma

    sz1 = U.shape
    if sz1[0] == 3:
        # For 3X3 matrices
        l_csl_p1 = zeros(3, 3)
        l_csl_p1[:, 0] = l_p1n_p1[:, 0]
        int_rat = (Rational(T0[1, 1]).limit_denominator())
        l_csl_p1[:, 1] = int_rat.p*l_p1n_p1[:, 1]
        l_csl_p1[:, 2] = T0[2, 2]*l_p1n_p1[:, 2]
        l1_csl_p1 = lt.lll_reduction(np.array(l_csl_p1, dtype='int64'))
        return Matrix(l1_csl_p1)
    # if sz1[0] == 2:
    #     ### For 2X2 matrices
    #     l_csl_p1 = zeros(2,2); l_csl_p1[:,0] = l_p1n_p1[:,0];
    #     int_rat = (Rational(T0[1,1]).limit_denominator())
    #     l_csl_p1[:,1] = int_rat.p*l_p1n_p1[:,1];
    #     return Matrix(l1_csl_p1);


def check_csl(l_csl_p, l_p1_po, T_p1top2_p1, Sigma, print_val):
    l_po_p1 = l_p1_po.inv()
    l_csl_po = l_p1_po*l_csl_p
    cond1 = int_man.check_int_mat(l_po_p1*l_csl_po, 1e-10)
    l_p2_p1 = Matrix(T_p1top2_p1)
    l_p2_po = l_p1_po*l_p2_p1
    l_po_p2 = l_p2_po.inv()
    cond2 = int_man.check_int_mat(l_po_p2*l_csl_po, 1e-10)
    Sigma1 = l_csl_po.det() / l_p1_po.det()
    cond3 = (np.abs(Sigma-Sigma1) < 1e-8)
    if print_val:
        if cond1:
            Disp_str = 'l_csl_po is defined in the l_p1_po lattice'
            print(Disp_str)
        if cond2:
            Disp_str = 'l_csl_po is defined in the l_p2_po lattice'
            print(Disp_str)
        if cond3:
            Disp_str = ('V(csl_po)/V(p1_po) = Sigma =  ' + "%0.0f" % (Sigma))
            print(Disp_str)
    return (cond1 and cond2 and cond3)


def check_dsc(l_dsc_p1, l_csl_p1, l_p1_po, l_p2_p1, Sigma, print_val):
    # l_po_p1 = l_p1_po.inv()
    l_csl_po = l_p1_po*l_csl_p1
    l_dsc_po = l_p1_po*l_dsc_p1
    l_p2_po = l_p1_po*l_p2_p1
    Tmat1 = l_dsc_po.inv()*l_p1_po
    cond1 = (int_man.check_int_mat(Tmat1, 1e-10))
    Tmat1 = l_dsc_po.inv()*l_p2_po
    cond2 = (int_man.check_int_mat(Tmat1, 1e-10))
    Tmat1 = l_dsc_po.inv()*l_csl_po
    cond3 = (int_man.check_int_mat(Tmat1, 1e-10))
    Tmat1 = l_dsc_p1*Sigma
    cond4 = (int_man.check_int_mat(Tmat1, 1e-10))
    Sigma1 = l_p1_po.det()/l_dsc_po.det()
    cond5 = (np.abs(Sigma-Sigma1) < 1e-8)
    if print_val:
        if cond1:
            Disp_str = 'l_p1_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond2:
            Disp_str = 'l_dsc_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond3:
            Disp_str = 'l_csl_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond4:
            Disp_str = 'l_dsc_po*Sigma is an integer matrix'
            print(Disp_str)
        if cond5:
            Disp_str = ('V(p1_po)/V(dsc_po) = Sigma =  ' + "%0.0f" % (Sigma))
            print(Disp_str)
    return (cond1 and cond2 and cond3 and cond4 and cond5)


def sigma_calc(t_g1tog2_g1, tol1):
    """
    Computes the sigma of the transformation matrix (t_g1tog2_g1)

    * Suppose T = t_g1tog2_g1
    * if det(T) = det(T^{-1}) then sigma1 = sigma2 is returned (homophase)
    * if det(T) \\neq det(T^{-1}) then max(sigma1, sigma2) is returned (heterophase)
    """
    R = Matrix(t_g1tog2_g1)
    R2 = R.det()*R.inv()
    Sigma21, _ = int_man.int_mult(R, tol1)
    Sigma22, _ = int_man.int_mult(R2, tol1)

    Sigma = int(np.array([Sigma21, Sigma22]).max())
    return int(Sigma)


def reciprocal_mat(l_g_go):
    """
    The reciprocal matrix with reciprocal basis vectors is computed for the
    input matrix with primitve basis vectors

    Parameters
    ----------------
    l_g_go: numpy array
        The primitive basis vectors b1x, b1y, b1z

    Returns
    -----------
    rl_g_go: numpy array
        The primitve reciprocal basis vectors
    """
    if isinstance(l_g_go, Matrix):
        InMat = np.array(l_g_go, dtype='double')
    else:
        InMat = np.array(l_g_go)

    L3 = np.cross(InMat[:, 0], InMat[:, 1]) / np.linalg.det(InMat)
    L1 = np.cross(InMat[:, 1], InMat[:, 2]) / np.linalg.det(InMat)
    L2 = np.cross(InMat[:, 2], InMat[:, 0]) / np.linalg.det(InMat)
    rl_g_go = np.vstack((L1, L2, L3)).T
    return Matrix(rl_g_go)


def make_right_handed(l_csl_p1, l_p_po):
    l_csl_po1 = l_p_po*l_csl_p1
    t1_array = np.array(l_csl_p1, dtype='double')
    t2_array = np.array(l_csl_p1, dtype='double')
    if (l_csl_po1.det() < 0):
        t1_array[:, 0] = t2_array[:, 1]
        t1_array[:, 1] = t2_array[:, 0]
    return Matrix(t1_array)
