import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#-- Utils: Functions that not need to build in class --#
def concatenate(data):
    """concatenate A function concatenate the trajectory data

    :param data: traj data that to be concatenated
    :type data: numpy array
    :return: concatenated data with shape (dim, n_traj * n_step)
    :rtype: numpy array
    """    
    cdata = data[0]
    for i in range(1, len(data)):
        cdata = np.concatenate((cdata, data[i]), axis=1)
    return cdata

def classifier(data, head_size=0, tail_size=10000, eps=0.5, visual_flag=False, debug = False):
    """classifier A classifier depend on DBSCAN to label different traj with labels wrt their position

    :param data: traj data to be labelled, shape as (n_traj, n_dim, n_step)
    :type data: numpy array
    :param head_size: head index of the data used for clustering, defaults to 0
    :type head_size: int, optional
    :param tail_size: end index of the data used for clustering, defaults to 10000
    :type tail_size: int, optional
    :param eps: parameter epsilon of DBSCAN, larger means the DBSCAN would consider a larger search for one class,
 defaults to 0.5
    :type eps: float, optional
    :param visual_flag: boolean value to indicate plot the labelling result or not, defaults to False
    :type visual_flag: bool, optional
    :return: corresponding labels of input trajectory data, shape as (n_traj,)
    :rtype: list
    """    
    # First Step: Take the data input, should consider the tail points
    data_tail = data[:,:,head_size:tail_size]
    data_len = tail_size - head_size
    X = concatenate(data_tail)
    # Second Step: apply DBSCAN and label all points
    clustering = DBSCAN(eps=eps, min_samples=2).fit(X.T)

    # Third Step: Assign label of points to its corresponding trajctory
    label=[]
    for i in range(1,len(X.T),data_len): # This should be modified wrt length of lables and length of assigned dataset in DBSCAN
        label.append(clustering.labels_[i])
    counts = [label.count(i) for i in set(label)]
    #DEBUG
    if debug:
        print(set(label))
        print(counts)
    lgd_idx = [label.index(i) for i in set(label)]
    # Optional: Plot of the result
    if visual_flag:
        for i in lgd_idx:
            plt.plot(data[i][0],data[i][1], color="C{}".format(label[i]), label=str(label[i]))
            plt.legend(loc='right')
        for i in range(len(data)):
            plt.plot(data[i][0], data[i][1], color="C{}".format(label[i]), alpha=0.2)
    # Output
    return label


def pickOut(data, labels, cnum=2, debug=False):
    """pickOut A atomic function for cleaning the data, picking out the outliner trajs that labelled as minority

    :param data: input raw data of trajs, not cleaned, shape as (n_traj, n_dim, n_step)
    :type data: numpy array
    :param labels: label of the raw data to be cleaned, shape as (ntraj,)
    :type labels: list
    :param cnum: number of class to be saved as the final output of cleaing, the top cnum number of labels would be kept,
defaults to 2
    :type cnum: int, optional
    :return: clean data, clean labels corresponding to the data
    :rtype: numpy array
    """
    assert len(data) == len(labels), "Data and Labels shoud have same dimension at axis 0"
    cls = list(set(labels)) # Given category of labels, defined as class of trajs
    counts = [labels.count(i) for i in cls]
    cls = [x for _,x in sorted(zip(counts,cls), reverse=True)]
    pData = []
    pLabels = []
    for idc in range(cnum):
        for idx in range(len(labels)):
            if labels[idx] == cls[idc]:
                pData.append(data[idx])
                pLabels.append(labels[idx])
            else:
                continue
    #DEBUG
    if debug:
        print(len(pData), len(pLabels))
    return np.array(pData), np.array(pLabels)
#-- End of Utils --#

#-- Class: DataProcess --#
class DataProcess:
    """ The data process class
    """    
    def __init__(self, debug=False):
        # Global flag params
        self.debug = debug
        # Label params
        self.head = 9900
        self.tail = 10000
        self.eps = 0.5
        # Clean params
        self.cnum = 2 # Modified at class function
        # Filter params
        self.fsize = 0 # Modified at class function

    def label(self, data, visual=False):
        return classifier(data, head_size=self.head, tail_size=self.tail,\
                          eps=self.eps, visual_flag=visual, debug = self.debug)

    def clean(self, data, labels, cnum=0, visual=False):
        # Record the class number input
        if cnum > 0: # indicating the user has changed the cnum themselves
            self.cnum=cnum
        else:
            pass # if the cnum keep 0 then use the modified cnum, default as 2
        clean_data, clean_labels = pickOut(data, labels, cnum=self.cnum, debug=self.debug)
        if visual:
            for traj, label in zip(clean_data, clean_labels):
                plt.plot(traj[0], traj[1], c= "C{}".format(label), alpha=0.5, marker='o', markersize=1, ls='')
        return  clean_data, clean_labels# in Data, labels order

    def filter(self, data, labels, fsize=10, visual=False):
        """filter A function to provide balanced data with each class in fsize number

        :param data: the trajectory data
        :type data: numpy array
        :param labels: the label corresponding to trajectory data
        :type labels: numpy array
        :param fsize: number of trajectories for each class in final result, defaults to 10
        :type fsize: int, optional
        :return: balanced data and balanced labels corresponding to the data
        :rtype: numpy.array numpy.array
        """        
        # Record the fileter size input
        self.fsize=fsize
        # Check index match
        labelSet = list(set(labels)) # get the set of labels
        assert len(labelSet) * self.fsize <= len(data), "Too few data, can not produce the required balanced dataset"
        labelCount = [list(labels).count(label) for label in labelSet]
        assert min(labelCount) > self.fsize, "The minimum data count{} cannot match requirement size"\
            .format(str(min(labelCount)))
        # Filter the data
        balanced_data = [] # Container of balance data
        balanced_labels = [] # Container of balanced labels
        sort_index = np.argsort(labels)
        Sdata = data[sort_index]
        Slabel = labels[sort_index]
        head = 0
        for i , c in zip(labelSet, labelCount):
            #DEBUG
            if self.debug:
                print("processing label " + str(i))
            assert(list(set(Slabel[head:head+self.fsize]))) == [i], "Index not match, please review the labels"
            balanced_labels.extend(Slabel[head:head+self.fsize])
            balanced_data.extend(Sdata[head:head+self.fsize])
            head += c
        if visual:
            for traj, label in zip(balanced_data, balanced_labels):
                plt.plot(traj[0], traj[1], c= "C{}".format(label), alpha=0.5, marker='o', markersize=1, ls='')
        return np.array(balanced_data), np.array(balanced_labels) # in Data, labels order


