import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class generateTraj:
    """ A class to define trajectory generator types, params and so on
    """    
    def __init__(self, dx, dy, type, help=False):
        """__init__ I/O to help define which potential the user want to generate lagenvin trajs on

        :param dx: Derivatives of potential for traj generation of x, user could either choose from utils or define own
        :type dx: function
        :param dy: Derivatives of potential for traj generation of y, user could either choose from utils or define own
        :type dy: function
        :param type: type of potential that would generate trajectories on, choose from spiral or zshape
        :type type: str
        :param help: boolean variable indicating should print the help message or not, defaults to False
        :type help: bool, optional
        """        
        #TODO : Check with pedro to see should put this in the doc only
        #TODO : Should add random seed I/O option for user to define so that the result could be repeated
        if help:
            print("Choose dx and dy from the following pairs:\n\
2-branch spiral: dx_s2, dy_s2\n\
3-branch spiral: dx_s3, dy_s3\n\
zshape: dx_z, dy_z\n")
            print("You can aslo modify your own potential functions by generating potential expressions and use them \
please refer to the documentation about two_D_pot potential generations")
        else:
            print("Some useful functions has been defined in ahead, set help=True for details")
        if type == 'spiral':
            self.n_steps = 10000
            self.n_dim = 2
            self.position_initial = [0.01, 0.01]
            self.simul_lagtime = 0.1
            self.friction = 10.0
            self.KbT = 0.5981
        elif type == 'zshape':
            self.n_steps = 10000
            self.n_dim = 2
            self.position_initial = [0., 1.0]
            self.simul_lagtime = 0.01
            self.friction = 5.0
            self.KbT = 0.5981
        self.dx = dx # Function of derivative of potential wrt to x
        self.dy = dy # Function of derivative of potential wrt to y
    def generate_traj(self):
        """generate_traj Generate single traj from the indicated potential, used as worker function when parallel

        :return: trajectory generated by Langevin algorithm
        :rtype: numpy array
        """        
        np.random.seed()
        #Initialize the traj container
        traj_langevin = np.zeros((self.n_dim, self.n_steps))
        traj_langevin[:,0] = self.position_initial # Transfer indices from matlab to python: -1 each, start from 0

        for step in range(1,self.n_steps):
            x_symb_val = traj_langevin[0, step-1] # Update values of x_symb for current position
            y_symb_val = traj_langevin[1, step-1]
            drift = np.dot(-1,[self.dx(x_symb_val,y_symb_val), self.dy(x_symb_val,y_symb_val)])
            traj_langevin[:, step] = traj_langevin[:, step-1] + drift * self.simul_lagtime/self.friction + \
                np.dot(np.random.randn(2,),np.sqrt(self.simul_lagtime*self.KbT/self.friction))
        return traj_langevin
    def batch_generator(self, batch_size = 100, visual=False):
        """batch_generator Generator function for producing batch of trajs, call the in-class generate function

        :param batch_size: number of trajectories to be generated at once, defaults to 100
        :type batch_size: int, optional
        :param visual: flag variable indicating should print the plot of trajs or not, defaults to False
        :type visual: bool, optional
        :return: A numpy array containing the generated traj data, shape as (batchSize, dimension, stepIndex)
        :rtype: numpy array
        """        
        # Generate Raw Data
        raw = []
        for i in tqdm(range(batch_size)):
            raw.append(self.generate_traj())
        raw = np.array(raw)
        if visual:
            for traj in raw:
                plt.scatter(traj[0], traj[1], s=1, alpha=0.6)
        return raw