import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def MLTSA(data, ans, model, drop_mode="Average", data_mode="Normal"):
    """

    Function to apply the Machine Learning Transition State Analysis to a given training dataset/answers and trained
    model. It calculates the Gloabl Means and re-calculates accuracy for predicting each outcome.

    :param data: Training data used for training the ML model. Must have shape (samples, features)
    :type data: list
    :param ans: Outcomes for each sample on "data". Shape must be (samples)
    :type ans: list
    :param model: Trained models used for prediction in the MLTSA process
    :type model: sklearn model class object
    :param drop_mode: flag indicating how to calculate the accuracy drop, optional, default as 'Average'
    :type drop_mode: str
    :param data_mode: flag indicating different input data structures, optional, default as 'Normal'
    :type data_mode: str
    :return: accuracy drop, shape as (n_features, n_samples)

    """

    if data_mode == "Rigged":
        data = data[:, :-1, :]

    # Calculating the global means
    means_per_sim = np.mean(data.T, axis=0) # (n_features, n_samples)
    gmeans = np.mean(means_per_sim, axis=1) # (n_features)
    temp_sim_data = np.copy(data) # (N_samples, n_features, n_steps)
    #DEBUG
    # print(temp_sim_data.shape)

    # Swapping the values and predicting for the FR
    FR = []
    for y, data in tqdm(enumerate(temp_sim_data)): # Looping through different samples
	# y: index of temp_sim_data for each data in temp_sim_data
	# data: One term of data in temp_sim_data array #TODO: Change 'data' to other names since it duplicates with input
        mean_sim = []
        for n, mean in enumerate(gmeans): # Looping through different features
	# n : index of gmeans for different features
	# mean: global mean of each feature
            tmp_dat = np.copy(data) # (n_features, n_steps)
	    #DEBUG
            # print(tmp_dat.shape)
            tmp_dat[n ,:] = mean # CORE ACTION: SWAPPING # Change the nth features to global mean value
	    #DEBUG
            # print(tmp_dat.T.shape)
            yy = model.predict(tmp_dat.T) # shape: (n_steps, 1)
	    # No need to decode here since do not need to encode the labels at first stage, sklearn do everything already.
            res = yy == ans[y] # Shape: (n_steps, 1)
            mean_sim.append(res) # At end of the loop, shape as (n_features, n_steps, 1)
        FR.append(mean_sim) # At end of the loop, shape as (n_samples, n_features, n_steps, 1)
    #DEBUG
    #print(np.array(FR).shape)

    #TODO: Fix the median, which is not in functionality now.
    if drop_mode == "Median":
        median = np.median(np.array(FR).T, axis=0)
        median = np.median(median, axis=0)
        dv_from_median = []
        for n, M in enumerate(median):
            dv_from_median.append(abs(M - np.array(FR)[n].T))
        fr = np.mean(np.array(dv_from_median), axis=1)
        fr = np.mean(fr, axis=0)


    if drop_mode == "Average":
        fr_per_sim = np.mean(np.array(FR).T, axis=0) # Shape: (n_steps, n_features, n_samples) (The dim 0 erased)
        fr = np.mean(fr_per_sim, axis=1) # Shape: (n_features, n_samples)

    return fr #(n_features, n_samples)


def MLTSA_Plot(FR, dataset_og, pots, errorbar=True):
    """

    Wrapper for plotting the results from the Accuracy Drop procedure

    :param FR: Values from the feature reduction
    :param dataset_og: Original dataset object class used for generating the data
    :param pots: Original potentials object class used for generating the data
    :param errorbar: Flag for including or not including the errobars in case of using replicas.
    :return:

    """
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

    #"""Code for the colorbar below"""
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
