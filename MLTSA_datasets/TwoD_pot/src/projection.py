import time
import numpy as np
from matplotlib import pyplot as plt


class Projector:
    """ Projector class which used for transformming of data, a method of resampling the data shape
    """    
    def __init__(self, n_features, rand_seed=False, keep_flag=True):
        """__init__ init method return a projector class

        :param n_features: number of features to be transformed, which indicates result number
        :type n_features: int
        :param rand_seed: the sampling random seed, defaults to False
        :type rand_seed: bool, optional, could be int to specify by user
        :param keep_flag: boolean value indicating to keep the origional traj data as a projection or not,\
defaults to True
        :type keep_flag: bool, optional
        """        
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
        """generator A function to generate the random coefficients as transform angle

        :return: angles which used for transformation
        :rtype: numpy array of float
        """        
        np.random.seed(self.random_seed)
        angles = np.random.uniform(0,2*np.pi,self.n_features)
        if self.seeding:
            angles[-1] = 0
        return angles     
    def rotation(self, traj, select = "X", plotting=False):
        """rotation Do transformation of the data, yield n_feature each time but only for one trajectory

        :param traj: single trajectory data to be transformed
        :type traj: numpy array, in shape of (n_dim, n_step)
        :param select: the sample value to be returned as new data, defaults to "X" which is X value of new trans data \
or "Y" which return Y value of new trans data as sample
        :type select: str, optional
        :param plotting: boolean value indicating plot the transformed result(the sample) or not, defaults to False
        :type plotting: bool, optional
        :return: features that sampled from transformed data, in case of select, result is featureX or featureY
        :rtype: numpy array, in shape of (n_feature, n_step)
        """        
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
        """batch_rotation SAME as before transforming 

        :param data: data containing all the trajectories to be transformed, in shape of (n_traj, n_dim, n_step)
        :type data: numpy array of float
        :param sel: selection of sample axis, same as rotation, defaults to "X"
        :type sel: str, optional
        :return: transformed and sampled result, depends on the sel, in shape of (n_traj, n_feature, n_step)
        :rtype: _type_
        """        
        projs = []
        for i in range(len(data)):
            projs.append(np.array(self.rotation(data[i], select=sel)))
        return np.array(projs)
    def show_axis(self):
        """show_axis visualisation function of the transformed new X axis
        """        
        for theta in self.coeff:
            temp = [np.cos(theta) * 2, np.sin(theta) * 2]
            plt.plot([0,temp[0]],[0,temp[1]])
            plt.arrow(0, 0, temp[0]/3, temp[1]/3, width=0.001, head_width=0.05, alpha=0.25, color='black')
        plt.xlim(-1,1)
        plt.ylim(-1,1)   
        plt.title("New positive x axis after rotation")
    def show_ax(self, data, labels, j):
        """show_ax showing the transformed new axis, but only for one parameter(one feature), plotting in front of the \
original data(trajectories)

        :param data: original trajectories data, in shape of (n_traj, n_dim, n_step)
        :type data: numpy array of float
        :param labels: original label of original data, in shape of n_traj
        :type labels: numpy array of int
        :param j: index of coeff to be selected from
        :type j: int
        """        
        for i in range(len(data)):
            plt.scatter(data[i][0], data[i][1], c="C{}".format(labels[i]),s=1)

        theta = self.coeff[j]
        temp = [np.cos(theta) * 13, np.sin(theta) * 13]
        plt.plot([0,temp[0]],[0,temp[1]], "--",  linewidth=3, color='black') 
        #ARROW
        ### axes = axis
        plt.arrow(0, 0, temp[0], temp[1], width=0.618,color='black', alpha=0.5)
    def show_rotation(self, data, labels, feature_id):
        """show_rotation A visualisation function to show the rotated trajectories, only to one feature

        :param data: original trajectories data, in shape of (n_traj, n_dim, n_step)
        :type data: numpy array of float
        :param labels: original label of original data, in shape of n_traj
        :type labels: numpy array of int
        :param feature_id: index of coeff to be selected from
        :type feature_id: int
        """        
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
        """show_transform A visualisation function showing the result of transform (whole trans traj) and the original \
trajectories

        :param data: original trajectories data, in shape of (n_traj, n_dim, n_step)
        :type data: numpy array of float
        :param labels: original label of original data, in shape of n_traj
        :type labels: numpy array of int
        :param feature_id: index of coeff to be selected from
        :type feature_id: int
        """           
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
        """show_feature A visualisation function to show sample of one feature, to all trajectories

        :param data: original trajectories data, in shape of (n_traj, n_dim, n_step)
        :type data: numpy array of float
        :param labels: original label of original data, in shape of n_traj
        :type labels: numpy array of int
        :param feature_id: index of coeff to be selected from
        :type feature_id: int
        :param sel: which sample of new transformed data to use, defaults to 'X' meaning use X axis value, could be "Y"
        :type sel: str, optional
        """        
        projs = self.batch_rotation(data, sel=sel)
        for proj, label in zip(projs[:,feature_id], labels):
                plt.plot(proj, color="C{}".format(label))
