import numpy as np
from matplotlib import pyplot as plt


class Projector:
    def __init__(self, n_features, seeding_flag=True):
        self.iter_time = n_features
        self.coefficients = np.zeros((n_features, 2))
        self.seeding_flag = seeding_flag

    def generator(self):
        a = np.random.uniform(-1, 1)  # [0,1) // a*cons1 + b*const2 a+b !~ uniform
        b = np.random.uniform(-1, 1)  # [0,1) * 2 -> [0,2) - 1 -> [-1,1)
        return a, b

    def project1D(self, X, Y, plotting=True):
        assert len(X) == len(Y), "X and Y should be in same shape"
        data = np.zeros((self.iter_time, len(X)))
        for i in range(self.iter_time):
            p1, p2 = self.generator()  # Generate paramters for each projection
            temp_data = p1 * np.array(X) + p2 * np.array(Y)
            self.coefficients[i][0] = p1
            self.coefficients[i][1] = p2
            # print(temp_data)
            data[i] = temp_data

        if self.seeding_flag:
            data[i] = np.array(X)  # Replace the last one to seed on X
            self.coefficients[i] = np.array([1, 0])

        if plotting == True:
            x = np.arange(min(X), max(X), 0.1)

            for i in range(self.iter_time):
                y = (-1 * self.coefficients[i][1] / self.coefficients[i][0]) * x
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.plot(x, y)

        return data



