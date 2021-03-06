import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD

if __name__ == '__main__':
    A = np.array([[1, 2, 3],
                  [3, 2, 5],
                  [5, 6, 1]])
    U, s, VT = svd(A)
    print(U)
    print(s)
    print(VT)
    print()


    Sigma = np.zeros(A.shape)
    for i in range(len(s)):
        Sigma[i][i] = s[i]
    print(Sigma)
    print(U.dot(Sigma).dot(VT))

    print(U[:, :2].dot(Sigma[:2, :2]).dot(VT[:2, :]))
