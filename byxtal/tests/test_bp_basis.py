#!/bin/bash

# Authors: Arash Dehghan Banadaki <adehgha@ncsu.edu>, Srikanth Patala <spatala@ncsu.edu>
# Copyright (c) 2015,  Arash Dehghan Banadaki and Srikanth Patala.
# License: GNU-GPL Style.
# How to cite GBpy:
# Banadaki, A. D. & Patala, S. "An efficient algorithm for computing the primitive bases of a general lattice plane",
# Journal of Applied Crystallography 48, 585-588 (2015). doi:10.1107/S1600576715004446


# -----------------------------------------------------------------------------------------------------------
### Write tests for gbpy.bp_basis functions



from sympy.matrices import Matrix, eye, zeros;
import numpy as np;

# import gbpy.integer_manipulations as int_man
L_p_po = Matrix(1.0 * np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]]));
# L_rp_po = fcd.reciprocal_mat(L_p_po);
# L_po_rp = (L_rp_po).inv();

# n_po = Matrix([2, 3, 1])
# n_rp = L_po_rp*n_po;


# nI_rp = int_man.int_finder(n_rp, 1e-10);

# # import gbpy.bp_basis as plb
# nI_rp1 = np.array(nI_rp, dtype='int64');
# L_planar_p = bp_basis(nI_rp1);

import byxtal
import byxtal.bp_basis as bpb

import os; import inspect; import pickle as pkl;
byxtal_dir = os.path.dirname((inspect.getfile(byxtal)))
pkl_dir = byxtal_dir + '/data_files'
pkl_file = pkl_dir + '/cF_Id_csl_common_rotations.pkl'
jar = open(pkl_file, 'rb');
pkl_content = pkl.load(jar);
jar.close();
sig_mats = pkl_content['sig_mats'];

Sigma3 = sig_mats['3_1'];

t_mat = Sigma3
index_type  = 'normal_go'
T_reference = 'g1'
# inds = np.array([2, 3, 1])
inds = np.array([4, 2, 2])

l_2d_csl_p1, l_pl1_p1, l_pl2_p1 = bpb.gb_2d_csl(inds, t_mat, L_p_po, index_type, T_reference)

bpb.check_2d_csl(l_pl1_p1, l_pl2_p1, l_2d_csl_p1)
