=======
Modules
=======

Lattice Class
-------------
.. autofunction:: byxtal.lattice.Lattice


Geometry Tools
---------------
.. autofunction:: byxtal.geometry_tools.sph2vec

.. autofunction:: byxtal.geometry_tools.idquaternion


Integer Manipulations
---------------------
.. autofunction:: byxtal.integer_manipulations.gcd_vec

.. autofunction:: byxtal.integer_manipulations.gcd_array

.. autofunction:: byxtal.integer_manipulations.lcm_vec

.. autofunction:: byxtal.integer_manipulations.lcm_array

.. autofunction:: byxtal.integer_manipulations.check_int_mat

.. autofunction:: byxtal.integer_manipulations.rat_approx

.. autofunction:: byxtal.integer_manipulations.int_approx

.. autofunction:: byxtal.integer_manipulations.int_mult_approx

.. autofunction:: byxtal.integer_manipulations.mult_fac_err

.. autofunction:: byxtal.integer_manipulations.int_finder

.. autofunction:: byxtal.integer_manipulations.int_check

.. autofunction:: byxtal.integer_manipulations.rat


CSL Utility Function
--------------------

.. autofunction:: byxtal.csl_utility_functions.proper_ptgrp

.. autofunction:: byxtal.csl_utility_functions.largest_odd_factor

.. autofunction:: byxtal.csl_utility_functions.compute_inp_params

.. autofunction:: byxtal.csl_utility_functions.mesh_muvw

.. autofunction:: byxtal.csl_utility_functions.mesh_muvw_fz

.. autofunction:: byxtal.csl_utility_functions.check_fsig_int

.. autofunction:: byxtal.csl_utility_functions.eliminate_idrots

.. autofunction:: byxtal.csl_utility_functions.sigtype_muvw

.. autofunction:: byxtal.csl_utility_functions.eliminate_mults

.. autofunction:: byxtal.csl_utility_functions.check_sigma

.. autofunction:: byxtal.csl_utility_functions.gcd1d_arr

.. autofunction:: byxtal.csl_utility_functions.compute_tmat

.. autofunction:: byxtal.csl_utility_functions.disorient_sigmarots

.. autofunction:: byxtal.csl_utility_functions.check_sigma_rots

.. autofunction:: byxtal.csl_utility_functions.csl_rotations

.. autofunction:: byxtal.csl_utility_functions.check_csl

Boundary Plane Basis
--------------------

.. autofunction:: byxtal.bp_basis.check_2d_csl

.. autofunction:: byxtal.bp_basis.lbi_dioph_soln

.. autofunction:: byxtal.bp_basis.compute_basis_vec

.. autofunction:: byxtal.bp_basis.bp_basis

.. autofunction:: byxtal.bp_basis.pl_density

.. autofunction:: byxtal.bp_basis.gb_2d_csl

.. autofunction:: byxtal.bp_basis.bicryst_planar_den


Misorientation Fundamental Zones
--------------------------------

.. autofunction:: byxtal.misorient_fz.check_cond

.. autofunction:: byxtal.misorient_fz.misorient_fz



Generate Symmetry Operators
----------------------------

.. autofunction:: byxtal.generate_symm_ops.generate_symm_mats

.. autofunction:: byxtal.generate_symm_ops.generate_symm_quats

.. autofunction:: byxtal.generate_symm_ops.save_symm_pkl


Tools
-----

.. autofunction:: byxtal.tools.unique_rows_tol

.. autofunction:: byxtal.tools.eq

.. autofunction:: byxtal.tools.message_display

.. autofunction:: byxtal.tools.extgcd

.. autofunction:: byxtal.tools.ehermite

.. autofunction:: byxtal.tools.left_matrix_division

.. autofunction:: byxtal.tools.smith_nf

.. autofunction:: byxtal.tools.vrrotvec2mat

.. autofunction:: byxtal.tools.vrrotmat2vec

.. autofunction:: byxtal.tools.quat2mat

.. autofunction:: byxtal.tools.mat2quat

.. autofunction:: byxtal.tools.axang2quat


Disorientation Symmetry Operation
----------------------------------
#.. autofunction:: byxtal.disorient_symm_props.disorient_symm_props


Find CSL and DSC
----------------

.. autofunction:: byxtal.find_csl_dsc.find_csl_dsc

.. autofunction:: byxtal.find_csl_dsc.csl_finder

.. autofunction:: byxtal.find_csl_dsc.dsc_finder

.. autofunction:: byxtal.find_csl_dsc.csl_finder_smith

.. autofunction:: byxtal.find_csl_dsc.check_csl

.. autofunction:: byxtal.find_csl_dsc.check_dsc

.. autofunction:: byxtal.find_csl_dsc.sigma_calc

.. autofunction:: byxtal.find_csl_dsc.reciprocal_mat

.. autofunction:: byxtal.find_csl_dsc.make_right_handed

lll-reduction
-------------
#.. autofunction:: byxtal.lll_tools.reduce_po_lat

#.. autofunction:: byxtal.lll_tools.check_basis_equi

#.. autofunction:: byxtal.lll_tools.check_basis_def

#.. autofunction:: byxtal.lll_tools.lll_reduction

#.. autofunction:: byxtal.lll_tools.mod

#.. autofunction:: byxtal.lll_tools.hermite_normal_form

#.. autofunction:: byxtal.lll_tools.smith_normal_form

#.. autofunction:: byxtal.lll_tools.row_col_all_zero

Pick Fundumental Zone
---------------------

.. autofunction:: byxtal.pick_fz_bpl.pick_fz_bpl

.. autofunction:: byxtal.pick_fz_bpl.rot_symm

Quaternion
----------
.. autofunction:: byxtal.quaternion.getq0

.. autofunction:: byxtal.quaternion.getq1

.. autofunction:: byxtal.quaternion.getq2

.. autofunction:: byxtal.quaternion.getq3

.. autofunction:: byxtal.quaternion.get_size

.. autofunction:: byxtal.quaternion.get_type

#.. autofunction:: byxtal.quaternion.display

.. autofunction:: byxtal.quaternion.antipodal

.. autofunction:: byxtal.quaternion.inverse

#.. autofunction:: byxtal.quaternion.mtimes

#.. autofunction:: byxtal.quaternion.eq

#.. autofunction:: byxtal.quaternion.quat2mat

#.. autofunction:: byxtal.quaternion.mat2quat

#.. autofunction:: byxtal.quaternion.ctranspose

3D Vector
---------