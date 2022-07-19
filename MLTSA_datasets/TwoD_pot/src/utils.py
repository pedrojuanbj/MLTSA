""" utils Functions of potential and derivatives, for use of generating, could be changed by advanced user

"""
import numpy as np
import numba as nb
from numpy import exp, sin, cos
from numpy import arctan2 as atan2

#-- 2D potential functions of 2-branch spiral potential --#
def potential_s2(x_symb, y_symb):
    return -(0.05*x_symb**2 + 0.05*y_symb**2 + 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**2)


@nb.njit()
def dx_s2(x_symb, y_symb):
    return -0.1*x_symb*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**2) - 1.02040816326531*(-0.2*x_symb*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) - 2*(0.04*x_symb + 2*y_symb/(x_symb**2 + y_symb**2))*(0.05*x_symb**2 + 0.05*y_symb**2)*cos(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)))*(-0.05*x_symb**2 - 0.05*y_symb**2 - 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**2)/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**3


@nb.njit()
def dy_s2(x_symb, y_symb):
    return -0.1*y_symb*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**2) - 1.02040816326531*(-0.2*y_symb*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) - 2*(0.05*x_symb**2 + 0.05*y_symb**2)*(-2*x_symb/(x_symb**2 + y_symb**2) + 0.04*y_symb)*cos(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)))*(-0.05*x_symb**2 - 0.05*y_symb**2 - 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**2)/(0.05*(x_symb**2 + y_symb**2)*sin(0.02*x_symb**2 + 0.02*y_symb**2 + 2*atan2(x_symb, y_symb)) + 1)**3
#-- End of 2D potential functions of 2-branch spiral potential

#-- 2D potential functions of 3-branch spiral potential --#
def potential_s3(x_symb, y_symb):
    return -(0.05*x_symb**2 + 0.05*y_symb**2 + 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**2)


@nb.njit()
def dx_s3(x_symb, y_symb):
    return -0.1*x_symb*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**2) - 1.02040816326531*(-0.2*x_symb*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) - 2*(0.06*x_symb + 3*y_symb/(x_symb**2 + y_symb**2))*(0.05*x_symb**2 + 0.05*y_symb**2)*cos(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)))*(-0.05*x_symb**2 - 0.05*y_symb**2 - 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**2)/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**3


@nb.njit()
def dy_s3(x_symb, y_symb):
    return -0.1*y_symb*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**2) - 1.02040816326531*(-0.2*y_symb*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) - 2*(0.05*x_symb**2 + 0.05*y_symb**2)*(-3*x_symb/(x_symb**2 + y_symb**2) + 0.06*y_symb)*cos(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)))*(-0.05*x_symb**2 - 0.05*y_symb**2 - 1)*exp(-1.02040816326531/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**2)/(0.05*(x_symb**2 + y_symb**2)*sin(0.03*x_symb**2 + 0.03*y_symb**2 + 3*atan2(x_symb, y_symb)) + 1)**3
#-- End of 2D potential functions of 3-branch spiral potential --#

#-- 2D potential functions of zshape potential --#
def potential_z(x_symb, y_symb):
    return exp(-x_symb - 2) + exp(x_symb - 2) + exp(-y_symb - 1) + exp(y_symb - 3) - 0.174819504261645*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - 9.66899354041083e-5*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - 0.31803826223279*exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + 4.53514739229025*x_symb - 2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - 1.43353817136578e-18*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - 6.89516776904996*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 2.00080032012806*x_symb - 12.5050020008003*y_symb**2 + 3.50140056022409*y_symb)


@nb.njit()
def dx_z(x_symb, y_symb):
    return -9.66899354041083e-5*(3.75 - 2.5*x_symb)*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - 0.174819504261645*(-2.5*x_symb - 3.75)*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - 0.31803826223279*(-5.66893424036281*x_symb - 4.53514739229025*y_symb + 4.53514739229025)*exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + 4.53514739229025*x_symb - 2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - 1.43353817136578e-18*(-4.0016006402561*x_symb + 7.00280112044818*y_symb - 12.0048019207683)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - 6.89516776904996*(-4.0016006402561*x_symb + 7.00280112044818*y_symb - 2.00080032012806)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 2.00080032012806*x_symb - 12.5050020008003*y_symb**2 + 3.50140056022409*y_symb) - exp(-x_symb - 2) + exp(x_symb - 2)


@nb.njit()
def dy_z(x_symb, y_symb):
    return -9.66899354041083e-5*(6.25 - 2.5*y_symb)*exp(-1.25*x_symb**2 + 3.75*x_symb - 1.25*y_symb**2 + 6.25*y_symb) - 0.174819504261645*(-2.5*y_symb - 1.25)*exp(-1.25*x_symb**2 - 3.75*x_symb - 1.25*y_symb**2 - 1.25*y_symb) - 0.31803826223279*(-4.53514739229025*x_symb - 5.66893424036281*y_symb + 5.66893424036281)*exp(-2.83446712018141*x_symb**2 - 4.53514739229025*x_symb*y_symb + 4.53514739229025*x_symb - 2.83446712018141*y_symb**2 + 5.66893424036281*y_symb) - 6.89516776904996*(7.00280112044818*x_symb - 25.0100040016006*y_symb + 3.50140056022409)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 2.00080032012806*x_symb - 12.5050020008003*y_symb**2 + 3.50140056022409*y_symb) - 1.43353817136578e-18*(7.00280112044818*x_symb - 25.0100040016006*y_symb + 46.5186074429772)*exp(-2.00080032012805*x_symb**2 + 7.00280112044818*x_symb*y_symb - 12.0048019207683*x_symb - 12.5050020008003*y_symb**2 + 46.5186074429772*y_symb) - exp(-y_symb - 1) + exp(y_symb - 3)
#-- End of 2D potential functions of zshape potential --#