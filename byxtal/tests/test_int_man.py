import byxtal.integer_manipulations as iman;

tau = 3/2.0;

tol1 = 1e-6;

n1, d1 = iman.rat(tau, tol1);


import numpy as np;


from sympy.matrices import Matrix, eye, zeros;

# test_str = 'rat_approx';
test_str = 'int_mult_irrat';

if test_str == 'int_mult_irrat':
    a1 = 3/np.sqrt(19);
    b1 = 2/np.sqrt(19);
    c1 = -5/np.sqrt(19);
    iarr1 = np.array([a1, b1, c1]);
    iarr2 = iman.int_finder(iarr1);

if test_str == 'rat_approx':
    tol1 = 1e-6;

    ct3 = 0;

    num1 = 100;
    diff_arr = np.zeros((num1,1))
    while ct3 < num1:
        rMat = np.random.rand(3,2);
        N1, D1 = iman.rat_approx(rMat, tol1);

        diff_mat = 0*rMat;
        sz = np.shape(rMat);
        for ct1 in range(sz[0]):
            for ct2 in range(sz[1]):
                diff_mat[ct1,ct2] = rMat[ct1,ct2] - N1[ct1, ct2]/D1[ct1, ct2];

        diff_arr[ct3] = np.max(np.abs(diff_mat));
        #print()
        ct3 = ct3 + 1;
    print(np.max(diff_arr))

