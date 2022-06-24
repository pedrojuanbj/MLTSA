import src
from src.generateTraj import generateTraj
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