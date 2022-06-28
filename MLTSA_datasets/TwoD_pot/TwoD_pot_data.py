import src
from src.generateTraj import generateTraj
from src.dataprocess import DataProcess
from src.projection import Projector
from src.utils import *
#-- Code for potential generation --#
def pot_generator(name):
    """pot_generator A getter function to get the object from potential generation class

    :param name: name of potential, spiral or zshape
    :type name: str
    :return: an object that contains metadata for generating potential expressions
    :rtype: self defined object
    """
    if name == 'spiral':
        obj = src.spiral.potential_spiral()
    elif name == 'zshape':
        obj = src.zshape.potential_zshape()
    print("Please check the attributes of generated objects, change any of the parameters before you use them")
    return obj

def show_pot_attributes(obj):
    """show_pot_attributes show the attributes of generated object, user could change those attributes before generation

    :param obj: obeject of potential class, yield by generate_potential_func above
    :type obj: object
    :return: attributes dictionary of self defined potential class, user could use attributes to assign value, however 
        this is not protected by type checking, so user should change the attributes very carefully and, only when they 
        are sure that the new input should work for potential generation.
    :rtype: dict
    """    
    return obj.__dict__

def get_pot_func(obj, return_flag=False):
    """get_pot_func a function to print the generated expressions for defining potential and derivatives

    :param obj: obeject of potential class, yield by generate_potential_func above, modified by user if they want
    :type obj: object
    :param return_flag: A boolean value indicating to save the expressions or not, defaults to False
    :type return_flag: bool, optional
    """    
    # Expressions templates
    POTENTIAL = """
def potential(x_symb, y_symb):
    return """
    DX = """
def dx(x_symb, y_symb):
    return """
    DY = """
def dy(x_symb, y_symb):
    return """
    # Get func str from obj
    pot, dx, dy = obj.generate_function()

    # Modify expressions
    POTENTIAL = POTENTIAL + pot
    DX = DX + dx
    DY = DY + dy

    # Return the expressions or print them for users to copy
    print("For best usage of the expressions, please use the following import command before running any of the \
functions: \n\nfrom numpy import exp, sin, cos\nfrom numpy import arctan2 as atan2")
    if return_flag:
        return POTENTIAL, DX, DY
    else:
        print(POTENTIAL + "\n\n")
        print(DX + "\n\n")
        print(DY + "\n\n")
#-- End of Code for potential generation --#

#-- Code for Traj generation --#
def generate_traj(type, number=100, visual=False, help=False):
    """generate_traj A meta function that use for generating traj data, simply wrapped I/O of gemerateTraj class

    :param type: pattern code of potential, indicating type of potential to generate traj on, including s2, s3 and z
    :type type: str
    :param number: number of trajs to be generated at once, defaults to 100
    :type number: int, optional
    :param visual: boolean value indicating plot the result of generated trajs or not, defaults to False
    :type visual: bool, optional
    :param help: boolean value indicating show the help information from generateTraj calss or not, defaults to False
    :type help: bool, optional
    :raises ValueError: indicating the user input not match pre-set potential parrtern code, should be z, s2 or s3

    :return: An array containing all the traj data in shape of (traj_index, dim_index(x or y), step_index)
    :rtype: numpy.array
    """    
    # Recognize pattern code, including spiral2(s2), spiral3(s3), zshape(z)
    if type == 's2':
        gen = generateTraj(dx_s2, dy_s2, 'spiral', help=help)
    elif type == 's3':
        gen = generateTraj(dx_s3, dy_s3, 'spiral', help=help)
    elif type == 'z':
        gen = generateTraj(dx_z, dy_z, 'zshape', help=help)
    else:
        raise ValueError('Pattern code not recognized, please choose from z, s2, s3 or\
modify the detailed trajectory generator by calling generateTraj class.')
    # Generate traj as required
    raw_trajs = gen.batch_generator(batch_size = number, visual=visual)
    return raw_trajs
#-- End of Code for traj generation --#

#-- Code for Data Process --#
def data_processor(type):
    """data_processor A function generate object of DataProcess class with respect to preset types

    :param type: type of raw trajectory data input, choose from 's2', 's3', 'z'
    :type type: str
    :return: DataProcess object with modified params
    :rtype: DataProcess object
    """    
    # Generate a dataProcess object for modification
    dp = DataProcess()
    # Depend on type, set the params of dp class
    if type == 's2':
        dp.eps = 2
    elif type == 's3':
        dp.eps = 1.5 # Based on test
        dp.cnum = 3
    elif type == 'z':
        pass # The defalut settings is perfect for zshape potential processing
    return dp

def data_process_full(obj, raw_data, fsize, visual=False):
    """data_process_full Process the generated raw traj data until balanced level

    :param obj: DataProcess obeject generated by data_processor function
    :type obj: DataProcess object
    :param raw_data: unprocessed traj data,  in shape of (n_traj, n_dim, n_step)
    :type raw_data: numpy array
    :param fsize: number of trajs in each class final produced, should be changed wrt raw_data length
    :type fsize: int
    :return: balance_data, balance_labels, processed data and corresponding labels
    :rtype: numpy array, numpy array
    """    
    # Get labels first
    raw_labels = obj.label(raw_data)
    # Clean data
    clean_data, clean_labels = obj.clean(raw_data, raw_labels)
    # Filter data
    balance_data, balance_labels = obj.filter(clean_data, clean_labels, fsize=fsize, visual=visual)
    return balance_data, balance_labels
#-- End of Code for Data Process --#

#-- Code for Data Projection --#
def data_projector(type, n_features=100):
    """data_projector A function to generate the projector object with respect to corresponding data potential type
       Including the following potential types: 's2', 's3' or 'z'
       Currently there is no difference for those potentials considering projectors

    :param type: type of potential that the data generates from, indicating how to tone the projector's param\
 automatically
    :type type: str
    :param n_features: number of features to be generated, defaults to 100
    :type n_features: int, optional
    :return: object of Projector Class
    :rtype: Class Projector
    """    
    if type == 's2':
        projector = Projector(n_features=n_features)
    elif type == 's3':
        projector = Projector(n_features=n_features)
    elif type == 'z':
        projector = Projector(n_features=n_features)
    return projector

def data_projection(obj, data):
    """data_projection a function produce projected data

    :param obj: projector generated from Projector class by function data_projector
    :type obj: Class Projector
    :param data: trajectory data, expected the balanced data from data_proecess
    :type data: numpy array
    :return: projs array, in shape as (n_trajs, n_features)
    :rtype: numpy array
    """    
    return obj.batch_rotation(data)