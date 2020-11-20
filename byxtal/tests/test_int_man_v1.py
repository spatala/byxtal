import numpy as np
import integer_manipulations as iman

####################################################
r1 = np.random.rand(8,2,6,8);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(5,8);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(10,1);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(10);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(1,10);
N1, D1 = iman.rat_approx(r1)
lcm1 = iman.lcm_arr(D1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

import pickle as pkl
pkl_name = 'vecs.pkl'
jar = open(pkl_name,'rb')
vecs_dict = pkl.load(jar)
jar.close()

vecs = vecs_dict['vecs']

sz1 = np.shape(vecs)[0]

diff_n = np.zeros((sz1,))
for ct1 in range(sz1):
    print(ct1)
    vec1 = vecs[ct1]
    u_vec1 = vec1/np.linalg.norm(vec1)
    i1, m1 = iman.int_approx(u_vec1, 1e-6)

    d_vec = i1-vec1
    if np.linalg.norm(d_vec) > 1e-10:
        ind1 = np.where(vec1 != 0)[0]
        m_arr = (np.unique(vec1[ind1]/i1[ind1]))
        m2 = m_arr[0]
        diff_n[ct1] = np.linalg.norm(i1*m2 - vec1)
    else:
        diff_n[ct1] = np.linalg.norm(vec1-i1)


# ct1 = 267
# print(ct1)
# vec1 = vecs[ct1]
# u_vec1 = vec1/np.linalg.norm(vec1)
# i1, m1 = iman.int_approx(u_vec1, 1e-6)

# d_vec = i1-vec1
# if np.linalg.norm(d_vec) > 1e-10:
#     ind1 = np.where(vec1 != 0)[0]
#     m_arr = (np.unique(vec1[ind1]/i1[ind1]))
#     m2 = m_arr[0]
#     print(np.linalg.norm(i1*m2 - vec1))
# else:
#     print(np.linalg.norm(vec1-i1))


