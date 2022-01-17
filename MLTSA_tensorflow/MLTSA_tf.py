import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def MLTSA(data, ans, model, mode=""):

    if mode == "Normal":
        data = data[:, :-1, :]

    # Calculating the global means
    means_per_sim = np.mean(data.T, axis=0)
    gmeans = np.mean(means_per_sim, axis=1)
    temp_sim_data = np.copy(data)
    # print(temp_sim_data.shape)

    # Swapping the values and predicting for the FR
    FR = []
    for y, data in tqdm(enumerate(temp_sim_data)):
        mean_sim = []
        for n, mean in enumerate(gmeans):
            tmp_dat = np.copy(data)
            # print(tmp_dat.shape)
            tmp_dat[n, :] = mean
            # print(tmp_dat.T.shape)
            yy = model.predict(tmp_dat.T)
            res = yy == ans[y]
            mean_sim.append(res)
        FR.append(mean_sim)
    fr_per_sim = np.mean(np.array(FR).T, axis=0)
    fr = np.mean(fr_per_sim, axis=1)
    return fr