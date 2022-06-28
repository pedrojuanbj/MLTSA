import time
import numpy as np
from matplotlib import pyplot as plt


class Projector:
    def __init__(self, n_features, rand_seed=False, keep_flag=True):
        self.n_features = n_features
        self.seeding = keep_flag
        if not rand_seed:
            self.random_seed = int(time.time())
            #print("Working")
        else:
            self.random_seed = rand_seed
            #print("Working manually")
        self.coeff = self.generator()
    def generator(self):
        np.random.seed(self.random_seed)
        angles = np.random.uniform(0,2*np.pi,self.n_features)
        if self.seeding:
            angles[-1] = 0
        return angles     
    def rotation(self, traj, select = "X", plotting=False):
        X = traj[0]
        Y = traj[1]
        featureX = []
        featureY = []
        for theta in self.coeff:
            sine = np.sin(theta)
            cosn = np.cos(theta)
            X_ = X * cosn + Y * sine
            Y_ = Y * cosn - X * sine
            featureX.append(X_)
            featureY.append(Y_)
        if plotting:
            self.show()
        if select == "X":
            return featureX
        elif select == "Y":
            return featureY
        else:
            print("Invalid indicated axis, accpet only X or Y")
    def batch_rotation(self, data, sel = "X"):
        projs = []
        for i in range(len(data)):
            projs.append(np.array(self.rotation(data[i], select=sel)))
        return np.array(projs)
    def show_axis(self):
        for theta in self.coeff:
            temp = [np.cos(theta) * 2, np.sin(theta) * 2]
            plt.plot([0,temp[0]],[0,temp[1]])
            plt.arrow(0, 0, temp[0]/3, temp[1]/3, width=0.001, head_width=0.05, alpha=0.25, color='black')
        plt.xlim(-1,1)
        plt.ylim(-1,1)   
        plt.title("New positive x axis after rotation")
    def show_ax(self, data, labels, j):
        for i in range(len(data)):
            plt.scatter(data[i][0], data[i][1], c="C{}".format(labels[i]),s=1)

        theta = self.coeff[j]
        temp = [np.cos(theta) * 13, np.sin(theta) * 13]
        plt.plot([0,temp[0]],[0,temp[1]], "--",  linewidth=3, color='black') 
        #ARROW
        ### axes = axis
        plt.arrow(0, 0, temp[0], temp[1], width=0.618,color='black', alpha=0.5)
    def show_rotation(self, data, labels, feature_id):
        projs_X = []
        projs_Y = []
        for i in range(len(data)):
            projs_X.append(np.array(self.rotation(data[i], 'X')))
            projs_Y.append(np.array(self.rotation(data[i], 'Y')))
        newX = np.array(projs_X)
        newY = np.array(projs_Y)
        f0X = newX[:,feature_id,:]
        f0Y = newY[:,feature_id,:]
        for i in range(len(data)):
            plt.scatter(f0X[i], f0Y[i], c="C{}".format(labels[i]),s=1)
    def show_transform(self, data, labels, feature_id):
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(121)
        ###
        for i in range(len(data)):
            ax1.scatter(data[i][0], data[i][1], c="C{}".format(labels[i]),s=1)

        theta = self.coeff[feature_id]
        temp = [np.cos(theta) * 13, np.sin(theta) * 13]
        #ax1.plot([0,temp[0]],[0,temp[1]], "--",  linewidth=3, color='black') 
        ax1.arrow(0, 0, temp[0], temp[1], width=0.001, head_width=1, alpha=0.5, color='black')
        ax1.set_xlim(-20,20)
        ax1.set_ylim(-20,20)
        ax1.set_title("Original Data")
        ###
        ax2 = plt.subplot(122)
        ###
        projs_X = []
        projs_Y = []
        for i in range(len(data)):
            projs_X.append(np.array(self.rotation(data[i], 'X')))
            projs_Y.append(np.array(self.rotation(data[i], 'Y')))
        newX = np.array(projs_X)
        newY = np.array(projs_Y)
        f0X = newX[:,feature_id,:]
        f0Y = newY[:,feature_id,:]
        for i in range(len(data)):
            ax2.scatter(f0X[i], f0Y[i], c="C{}".format(labels[i]),s=1)
        ax2.set_xlim(-20,20)
        ax2.set_ylim(-20,20)
        ax2.set_title("Transformed Data")
        ###
        plt.axis("equal")
    def show_feature(self, data, labels, feature_id, sel='X'):
        projs = self.batch_rotation(data, sel=sel)
        for proj, label in zip(projs[:,feature_id], labels):
                plt.plot(proj, color="C{}".format(label))
