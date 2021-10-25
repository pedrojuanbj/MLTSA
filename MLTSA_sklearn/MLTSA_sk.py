import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def MLTSA(data, ans, model, mode):

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
            tmp_dat[n ,:] = mean
            # print(tmp_dat.T.shape)
            yy = model.predict(tmp_dat.T)
            res = yy == ans[y]
            mean_sim.append(res)
        FR.append(mean_sim)
    fr_per_sim = np.mean(np.array(FR).T, axis=0)
    fr = np.mean(fr_per_sim, axis=1)
    return fr


def MLTSA_Plot(FR, dataset_og, pots, errorbar=True):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl
    # Fetching info
    std = np.std(FR, axis=0, ddof=1) * 100
    dat = np.mean(FR, axis=0) * 100
    coefs = dataset_og.mixing_coefs
    coefs = np.array(coefs).T
    combs = dataset_og.combinations
    imp_id = pots.relevant_id

    # Getting the correlated features
    cor_feats = []
    for n, idx in enumerate(combs[:len(dat)]):
        if imp_id in idx:
            cor_feats.append(n)

    # Calculating Correlation relation
    correlations = []
    for n, f in enumerate(cor_feats):
        correlation = coefs[f][np.where(np.array(combs[f]) == imp_id)[0][0]] / np.sum(coefs[f]) * 100
        correlations.append(correlation)

    plt.figure(figsize=(10, 3))
    plt.title("Feature Reduction")
    plt.plot(dat, "-o", color="black", marker="s")
    # print(dat)
    #     if errorbar == True:
    #         plt.errorbar(np.arange(0, len(dat)), dat, yerr=std,
    #                      capsize=5, linestyle="None", marker="^", color="black")

    for correlated_feat, corr in zip(cor_feats, correlations):
        # rgb = colorsys.hsv_to_rgb((200+int(corr))/300., 1.0, 1.0)
        rgb = ((corr * 2.55) / 255, 0, 1 - (corr * 2.55) / 255)
        plt.plot(correlated_feat, dat[correlated_feat], "X", markersize=10, color=rgb)
        if errorbar == True:
            plt.errorbar(correlated_feat, dat[correlated_feat], yerr=std[correlated_feat],
                         capsize=5, linestyle="None", marker="^", color="black")
        #         plt.annotate("{:.2f}".format(corr),
        #                      (correlated_feat, dat[correlated_feat]),
        #                     fontsize=12, fontstyle="italic", fontweight="medium")
        if corr > 30:
            plt.text(correlated_feat + 6, dat[correlated_feat] - 0.005, "{:.2f}".format(corr),
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                     fontstyle="italic", fontweight="medium"
                     )
    plt.ylabel("Accuracy (%)")
    plt.xlabel("#Feature")
    # plt.colorbar()

    """Code for the colorbar below"""
    #print("Plotting Colorbar")
    rgb1 = (0, 0, 1)
    rgb2 = (1, 0, 0)
    cmap_name = "corr"
    cmap = LinearSegmentedColormap.from_list(cmap_name, [rgb1, rgb2], N=100)
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation='horizontal', label='Correlation to DW potential (%)')
    #fig.show()

    return
