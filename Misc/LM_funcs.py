
import os
import sys


import numpy as np
from numba import vectorize, njit, jit, prange, guvectorize 

 
import pandas as pd


import utils
from het_block import het
from simple_block import simple
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import ticker

plt.style.use('ggplot')


#%config InlineBackend.figure_format = 'retina'

from scipy.interpolate import interp2d
from scipy.optimize import minimize 
from solved_block import solved
from scipy import optimize

import pickle
import scipy.interpolate  
from scipy import stats
#import ineqpy
from scipy.ndimage.interpolation import shift


params = {'legend.fontsize': 'small',
          'figure.figsize': (7, 5),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small',
         'figure.frameon' : 'True',
         'axes.edgecolor' : 'black'}

from tabulate import tabulate


from scipy.stats import norm
import jacobian as jac
import nonlinear
import determinacy as det

from consav.misc import  nonlinspace 

from statsmodels.nonparametric.kde import KDEUnivariate # weighted kernel density 

#from types import SimpleNamespace

#from consav import upperenvelope, runtools 
#runtools.write_numba_config(disable=0,threads=4)


import FigUtils  

from quantecon import lorenz_curve

from Utils2 import *

from Models import *


# LM functions

def LM_models_shock(Ivar):  
    exogenous = [Ivar]
    settings = {'save' : False, 'use_saved' : True, 'Fvac_share' : False}
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = False
    settings['vac2w_costs'] = 0.05 
    settings['Fvac_factor'] = 5
    settings['destrO_share'] = 0.5 
    
    settings['SAM_model'] = 'Standard'
    ss_standard = ss_calib('Solve', settings)     
    
    block_list=[laborMarket1, laborMarket2, firm_labor_standard, wage_func1, ProdFunc]
    unknowns = ['N', 'S', 'Tight',  'JV', 'JM', 'Y']
    targets = [ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res', 'ProdFunc_Res']
    
    Time = 300 
    G_jac_standard = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_standard, save=False)
    
    
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'simple'
    ss_costly = ss_calib('Solve', settings)
    
    block_list = [laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wage_func1, ProdFunc]
    unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y']
    targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res']   
        
    G_jac_costly_vac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly, save=False)
    
    # FR 
    if Ivar == 'destr':
        exogenous = ['destrO']
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    ss_costly_FR = ss_calib('Solve', settings)
    
    block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate]
    unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y']
    targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res']   
        
    G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
    
    G_jac_costly_vac_FR_destrNO = G_jac_costly_vac_FR
    if Ivar == 'destr':
        exogenous = ['destrNO']
        G_jac_costly_vac_FR_destrNO = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)

    return ss_standard, ss_costly, ss_costly_FR, G_jac_standard, G_jac_costly_vac,  G_jac_costly_vac_FR, G_jac_costly_vac_FR_destrNO


def LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock):
    shocklength = 10
    dZ_PE = np.zeros([Time])
    dZ_PE[:shocklength] =  - 0.01  * ss[IvarPE]
    dZ_GE = np.zeros([Time])
    dZ_GE[:shocklength] =  GE_shock
    return dZ_PE, dZ_GE