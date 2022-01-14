import numpy as np
import matplotlib.pyplot as plt
import MLTSA_datsets.OneD_pot.OneD_pot_data as oned
from sklearn.preprocessing import OneHotEncoder

import MLTSA_datsets.OneD_pot.OneD_pot_data

if __name__ == '__main__':

    pots = oned.potentials(10, 1, 0)
    oneD_dataset = oned.dataset(pots, 100, 2)
    print("Dataset created")
    print("Generating trajectories")

    data, ans = oneD_dataset.generate_linear(100, 500)
    data_val, ans_val = oneD_dataset.generate_linear(25, 500)

    print("Preparing data for ML")
    time_frame = [30, 60] #Same time frame as the sklearn one
    X, Y = oneD_dataset.PrepareData(data, ans, time_frame)
    X_val, Y_val = oneD_dataset.PrepareData(data_val, ans_val, time_frame)

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(len(Y), 1)).toarray()
    Y_val = encoder.fit_transform(Y_val.reshape(len(Y_val),1)).toarray()
    print(Y)
    print(Y_val)


    