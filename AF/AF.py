# %%

from math import sqrt
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# %%
class AverageFilling:
    def __init__(self, records_train, records_test):
        records = np.vstack([records_train, records_test])
        n = len(np.unique(np.sort(records[:, 0])))
        m = len(np.unique(np.sort(records[:, 1])))

        # Initial R
        R = np.zeros([n, m])
        for record in records_train:
            R[record[0], record[1]] = record[2]

        # Initial indicator
        y = np.where(R, 1, 0)
        y_user = np.sum(y, axis=1)
        y_item = np.sum(y, axis=0)

        # Global average of rating
        self.r = np.sum(R) / np.sum(y)

        # average rating of user
        self.r_u = np.where(y_user,
                            np.sum(R, axis=1) / y_user,
                            self.r)

        # average rating of item
        self.r_i = np.where(y_item,
                            np.sum(R, axis=0) / y_item,
                            self.r)

        # bias of user
        self.b_u = np.where(y_user,
                            np.sum(y * (R - self.r_i), axis=1) / y_user,
                            0)

        # bias of item
        self.b_i = np.where(y_item,
                            np.sum(y * (R - self.r_u.reshape(-1, 1)), axis=0) / y_item,
                            0)

        # user segmentation based on activeness
        self.user_segmentation(n, m, records_test)

    def user_segmentation(self, n, m, records_test):
        R = np.zeros([n, m])
        for record in records_test:
            R[record[0], record[1]] = record[2]

        # Initial indicator
        y = np.where(R, 1, 0)
        y_user = np.sum(y, axis=1)

        # Segmentation
        Groups = []
        Groups.append(np.where((y_user <= 20) * (y_user > 0)))
        Groups.append(np.where((y_user <= 50) * (y_user > 20)))
        Groups.append(np.where((y_user > 50)))
        # self.ratings_grouptest = [np.vstack([records_test[tuple([records_test[:, 0] == user])]
        #                          for user in group[0]]) for group in Groups]

        self.ratings_grouptest = [np.vstack([records_test[records_test[:, 0] == user]
                                             for user in group[0]]) for group in Groups]


    def performance_on_user_segmentation(self):
        return np.vstack([self.performance(rating_grouptest) for rating_grouptest in self.ratings_grouptest])

    def performance(self, records_test):
        return [self.user_average(records_test), self.item_average(records_test),
                self.mean_of_user_item_average(records_test), self.user_bias_item_average(records_test),
                self.user_average_item_bias(records_test), self.global_average_user_bias_item_bias(records_test)]

    def user_average(self, records_test):
        return self.score(self.r_u[records_test[:, 0]], records_test[:, 2])

    def item_average(self, records_test):
        return self.score(self.r_i[records_test[:, 1]], records_test[:, 2])

    def mean_of_user_item_average(self, records_test):
        return self.score((self.r_i[records_test[:, 1]] + self.r_u[records_test[:, 0]]) / 2,
                          records_test[:, 2])

    def user_bias_item_average(self, records_test):
        return self.score(self.b_u[records_test[:, 0]] + self.r_i[records_test[:, 1]], records_test[:, 2])

    def user_average_item_bias(self, records_test):
        return self.score(self.r_u[records_test[:, 0]] + self.b_i[records_test[:, 1]], records_test[:, 2])

    def global_average_user_bias_item_bias(self, records_test):
        return self.score(self.b_u[records_test[:, 0]] + self.b_i[records_test[:, 1]] + self.r, records_test[:, 2])

    def score(self, rating_test, rating_predict):
        return [round(sqrt(metrics.mean_squared_error(rating_test, rating_predict)), 4),
                round(metrics.mean_absolute_error(rating_test, rating_predict), 4)]


if __name__ == '__main__':
    # Load data
    records_train = np.loadtxt('../data/ml-100k/u1.base', dtype=np.int32)
    records_test = np.loadtxt('../data/ml-100k/u1.test', dtype=np.int32)
    records_train[:, :2] -= 1
    records_test[:, :2] -= 1
    rating_test = records_test[:, 2]

    # Declare an Average Filler
    af = AverageFilling(records_train, records_test)

    # Performance
    af.user_average(records_test)
    af.item_average(records_test)
    af.mean_of_user_item_average(records_test)
    af.user_bias_item_average(records_test)
    af.user_average_item_bias(records_test)
    af.global_average_user_bias_item_bias(records_test)




