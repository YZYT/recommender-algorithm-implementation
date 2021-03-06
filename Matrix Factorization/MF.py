
from math import sqrt
from operator import itemgetter

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import metrics


class PureSingularValueDecomposition:
    def __init__(self, records_train, records_test):
        records = np.vstack([records_train, records_test])
        self.n = len(np.unique(np.sort(records[:, 0])))
        self.m = len(np.unique(np.sort(records[:, 1])))

        # Initial R
        self.R = np.zeros([self.n, self.m], dtype=np.int32)

        for record in records_train:
            self.R[record[0], record[1]] = record[2]

        # Initial indicator
        y = np.where(self.R, 1, 0)
        y_user = np.sum(y, axis=1)
        y_item = np.sum(y, axis=0)

        # Global average of rating
        self.r = np.sum(self.R) / np.sum(y)

        # average rating of user
        self.r_u = np.where(y_user,
                            np.sum(self.R, axis=1) / y_user,
                            self.r)

        # average rating of item
        self.r_i = np.where(y_item,
                            np.sum(self.R, axis=0) / y_item,
                            self.r)

        # filling the matrix
        self.R = np.where(self.R == 0,
                          np.zeros(shape=self.R.shape) + self.r_u.reshape(-1, 1),
                          self.R)

        # SVD
        U, s, VT = svd(self.R)

        d = 20
        Sigma = np.zeros([d, d])
        for i in range(d):
            Sigma[i][i] = s[i]

        self.R = U[:, :d].dot(Sigma).dot(VT[:d, :])

    # def performance(self, records_test):
    #     # print(self.R[:5, :5])
    #     return self.R[records_test[:, 0], records_test[:, 1]] + self.r_u[records_test[:, 0]]

    def performance(self, records_test):
        # print(self.R[:5, :5])
        return self.R[records_test[:, 0], records_test[:, 1]]

