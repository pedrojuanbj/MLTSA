import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as spy
from numpy import exp, sin, cos, tan
from numpy import arctan2 as atan2

class potential_zshape:
    """ zshape potential generation class, including 3 example expressions of zshape potential with derivatives
    """    
    def __init__(self,IN_n_states=[200,200]):
        """__init__ initializing the zshape class

        :param IN_n_states: number of states as input, defaults to [200,200]
        :type IN_n_states: list, optional
        """         
        x_min = -2
        x_max = 2
        y_min = -1
        y_max = 3
        #---Hyper-parameters---#
        border_coeff = 1; # Coefficient of borders 
        well_coeff = 10; # Coefficient of wells (larger value makes wells deeper and barrier between them bigger)
        
        #---PARAMETERS OF NORMAL DISTRIBUTIONS---#
        # Section "Bivariate case" in [https://en.wikipedia.org/wiki/Multivariate_normal_distribution]
        mu_1 = np.array([-1.5,-0.5]).reshape(2,1)
        mu_2 = np.array([1.5,2.5]).reshape(2,1)
        sigm_center_well = 0.4 # Sigma of some of the wells, sigm for sigma
        covar_mat_well = np.diag(np.dot(np.ones((len(IN_n_states))),sigm_center_well))
        ###Debugging
        #print(covar_mat_well)

        mu_3 = np.array([-0.5, 0]).reshape(2,1)
        sig3 = np.array([0.7, 0.28]).reshape(2,1) # Origion: [1, 0.4] * 0.7 = result, flat one
        ro3 = 0.7
        colvar_mat_3 = np.array([[sig3[0]**2, ro3*sig3[0]*sig3[1]],[ro3*sig3[0]*sig3[1], sig3[1]**2]]).reshape(2,2)

        mu_4 = np.array([0, 1]).reshape(2,1)
        sig4 = np.array([0.7, 0.7]).reshape(2,1) # Origion: [1, 1] * 0.7, Diagonal one
        ro4 = -0.8
        colvar_mat_4 = np.array([[sig4[0]**2, ro4*sig4[0]*sig4[1]],[ro4*sig4[0]*sig4[1], sig4[1]**2]]).reshape(2,2)

        mu_5 = np.array([0.5, 2]).reshape(2,1)
        sig5 = np.array([0.7, 0.28]).reshape(2,1) # Origion: [1, 0.4] * 0.7 = result, flat one
        ro5 = 0.7
        colvar_mat_5 = np.array([[sig5[0]**2, ro5*sig5[0]*sig5[1]],[ro5*sig5[0]*sig5[1], sig5[1]**2]]).reshape(2,2)

        # Symbolic Settings #
        x_symb, y_symb = spy.symbols('x_symb, y_symb', real=True)
        vec_symb = spy.Matrix([x_symb,y_symb])
        # BORDERS (Potential increase to infinity outside of [x_min,x_max] and [y_min,y_max])
        b_1 = spy.exp(x_min - x_symb)
        b_2 = spy.exp(x_symb - x_max)
        b_3 = spy.exp(y_min - y_symb)
        b_4 = spy.exp(y_symb - y_max)
        w_1 = ((2*np.pi)**(-1))*(np.linalg.det(covar_mat_well)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_1).T*\
            np.linalg.inv(covar_mat_well)*(vec_symb-mu_1))

        # WELLS
        w_1 = ((2*np.pi)**(-1))*(np.linalg.det(covar_mat_well)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_1).T*\
                np.linalg.inv(covar_mat_well)*(vec_symb-mu_1))
        w_2 = ((2*np.pi)**(-1))*(np.linalg.det(covar_mat_well)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_2).T*\
                np.linalg.inv(covar_mat_well)*(vec_symb-mu_2))
        
        w_3 = ((2*np.pi)**(-1))*(np.linalg.det(colvar_mat_3)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_3).T*\
                np.linalg.inv(colvar_mat_3)*(vec_symb-mu_3))
        
        w_4 = ((2*np.pi)**(-1))*(np.linalg.det(colvar_mat_4)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_4).T*\
                np.linalg.inv(colvar_mat_4)*(vec_symb-mu_4))
        w_5 = ((2*np.pi)**(-1))*(np.linalg.det(colvar_mat_5)**(-1/2))*spy.exp((-1/2)*(vec_symb-mu_5).T*\
                np.linalg.inv(colvar_mat_5)*(vec_symb-mu_5))
        
        # Border & Well Construction
        border_constribution_symbolic = (b_1 + b_2 + b_3 + b_4)*border_coeff
        well_contribution_symbolic = (w_1[0] + w_2[0] + w_3[0] + w_4[0] + w_5[0])*well_coeff

        # Build of Potential Symbolic
        self.potential_symbolic = border_constribution_symbolic - well_contribution_symbolic
        
        # Symbolic Derivatives with respect to x and y
        self.dpotsym_dx = self.potential_symbolic.diff(x_symb)
        self.dpotsym_dy = self.potential_symbolic.diff(y_symb)

    def generate_function(self):
        """Generating the expression of potential function and corresponding derivatives
        """
        # Get the equations of potential and derivatives
        spiral_pot = str(self.potential_symbolic)
        spiral_dx = str(self.dpotsym_dx)
        spiral_dy = str(self.dpotsym_dy)
        print("Return in order of: potential, dx, dy")
        return spiral_pot, spiral_dx, spiral_dy

    def visualize_3d(self):
        """Plotting the 3d surface of zshape, potential provided by example function
        """
        # Meshgrid of potential surface
        [x,y] = np.meshgrid(np.linspace(-2, 2, 200),np.linspace(-1, 3, 200))

        # Calculatin Energy for each
        z = self.expl_potential(x, y)

        #Plotting the potential Surface
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, z, cmap=plt.cm.viridis, alpha=0.6)
        

    def expl_potential(self, x_symb, y_symb):
        """An Example of 2d, zshape potential function generated by symbolic expressions,\
             used for visualisation

        :param x_symb: x input
        :type x_symb: float
        :param y_symb: y input 
        :type y_symb: float
        :return: potential value at x,y
        :rtype: float
        """        
        return exp(-x_symb - 2) + exp(x_symb - 2) + exp(-y_symb - 1) + exp(y_symb - 3) - \
            0.174819504261645*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - \
            9.66899354041083e-5*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - \
            0.31803826223279*exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + \
            4.53514739229025*x_symb - 2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - \
            1.43353817136578e-18*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - \
            12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - \
            6.89516776904996*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - \
            2.00080032012806*x_symb - 12.5050020008003*y_symb**2 + 3.50140056022409*y_symb)

    def expl_dx(x_symb, y_symb):
        """Example derivative of potential, wrt to x

        :param x_symb: x input
        :type x_symb: float
        :param y_symb: y input
        :type y_symb: float
        :return: derivative of x value at x,y
        :rtype: float
        """        
        print("Function not accelerated by numba, only used for limited visualisation,\
               to produce data pipeline please refer to other methods")
        return -9.66899354041083e-5*(3.75 - 2.5*x_symb)*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - \
            0.174819504261645*(-2.5*x_symb - 3.75)*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - \
            0.31803826223279*(-5.66893424036281*x_symb - 4.53514739229025*y_symb + 4.53514739229025)*\
            exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + 4.53514739229025*x_symb - \
            2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - 1.43353817136578e-18*(-4.0016006402561*x_symb + \
            7.00280112044818*y_symb - 12.0048019207683)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb \
            - 12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - \
            6.89516776904996*(-4.0016006402561*x_symb + 7.00280112044818*y_symb - 2.00080032012806)*\
            exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 2.00080032012806*x_symb - \
            12.5050020008003*y_symb**2 + 3.50140056022409*y_symb) - exp(-x_symb - 2) + exp(x_symb - 2)

    def expl_dy(x_symb, y_symb):
        """Example derivative of potential, wrt to y
        
        :param x_symb: x input
        :type x_symb: float
        :param y_symb: y input
        :type y_symb: float
        :return: value of derivative of y value at point x,y
        :rtype: float
        """        

        print("Function not accelerated by numba, only used for limited visualisation,\
               to produce data pipeline please refer to other methods")
        return -9.66899354041083e-5*(6.25 - 2.5*y_symb)*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - \
            0.174819504261645*(-2.5*y_symb - 1.25)*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - \
            0.31803826223279*(-4.53514739229025*x_symb - 5.66893424036281*y_symb + 5.66893424036281)*\
            exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + 4.53514739229025*x_symb - \
            2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - 6.89516776904996*(7.00280112044818*x_symb - \
            25.0100040016006*y_symb + 3.50140056022409)*exp(-2.00080032012805*x_symb**2 + \
            7.00280112044818*x_symb*y_symb - 2.00080032012806*x_symb - 12.5050020008003*y_symb**2 + \
            3.50140056022409*y_symb) - 1.43353817136578e-18*(7.00280112044818*x_symb - 25.0100040016006*y_symb + \
            46.5186074429772)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - \
            12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - \
            exp(-y_symb - 1) + exp(y_symb - 3)