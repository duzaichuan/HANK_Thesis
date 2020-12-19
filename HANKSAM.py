
import os
import sys


os.chdir(os.path.dirname(os.path.abspath(__file__)))

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



'''Part 1: HA block'''
@het(exogenous='Pi', policy=['a'], backward=['EVa'])
def EGMhousehold( EVa_p, Pi_p, Pi_ern, a_grid, Pi_seed, rstar, sBorrow, P, hss, kappa, ssN,
                 e_grid,pi_e, pi_ern, w, ra, beta, eis, Eq, N_, destr_L, destrO_L, b, Benefit_type, T_dist, pi_beta, wss,
                 Tss, Ttd, Tuni, ssAvgInc, VAT,  nPoints, cs_avg, ttd_inf, frisch, ssflag=False):
    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""


     
    x = a_grid >= 0
    x = x.astype(int)
    R = 1 + ra * x + (1- x) * (ra + kappa)    
    
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2
    
    e_grid_alt = e_grid
        
    sol = {'N' : {}, 'S' : {}} # create dict containing vars for each employment state 
    
    # reshape some input
    EVa_p  =                 np.reshape(EVa_p, (nBeta, ne, nN, nA))
    U_inc, _  = Unemp_benefit(Benefit_type, b, e_grid, pi_ern, ne, wss) 
    

    
    T =  transfers(pi_ern, Tss, e_grid, T_dist)

    Ttd_ = transfers_td_e(pi_ern, Ttd, e_grid)
    
    T_agg =  np.broadcast_to(T[np.newaxis, :, np.newaxis] + Ttd_[np.newaxis, :, np.newaxis] + Tuni , (nBeta,ne, nA)) 
    
    # u'c(z_t, a_t) 
    sol['N']['uc_nextgrid']  = np.ones([nBeta, ne, nA]) * np.nan
    sol['S']['uc_nextgrid']  = np.ones([nBeta, ne, nA]) * np.nan
    sol['N']['tInc']         = np.ones([nBeta, ne, nA]) * np.nan
    sol['S']['tInc']         = np.ones([nBeta, ne, nA]) * np.nan


    a_grid_org = a_grid
    dN = N_ / ssN
    dU = (1-N_) / (1-ssN)
    
    for j in range(nBeta):
        sol['N']['uc_nextgrid'][j,:,:]  =  (beta[j] * Pi_ern) @ EVa_p[j,:,0, :] 
        sol['S']['uc_nextgrid'][j,:,:]  =  (beta[j] * Pi_ern) @ EVa_p[j,:,1, :] 
            
    for h in sol:    
        if h == 'N':
            a_grid = a_grid_org 
        if h == 'S':
            a_grid = a_grid_org 
            
        sol[h]['c_nextgrid'] = inv_mu(sol[h]['uc_nextgrid'], eis)
        if h == 'N':
            tax_IN  = avgTaxf(w   * e_grid, ssAvgInc, cs_avg) 
            sol[h]['tInc']  = np.broadcast_to(tax_IN[np.newaxis, :, np.newaxis], (nBeta,ne, nA))
            sol[h]['I']   =   (1 - sol[h]['tInc'])  * w   * e_grid[np.newaxis, :, np.newaxis] * dN + T_agg
        if h == 'S': 
            tax_IS  = avgTaxf(U_inc, ssAvgInc, cs_avg) 
            sol[h]['tInc']  = np.broadcast_to(tax_IS[np.newaxis, :, np.newaxis], (nBeta,ne, nA))
            sol[h]['I']   =   (1 - sol[h]['tInc'])  * U_inc[np.newaxis,:, np.newaxis]  * dU  + T_agg

        # interpolate for each beta 
        sol[h]['c'] =  np.empty([nBeta, ne, nA])  
        

        for j in range(nBeta):
            lhs = sol[h]['c_nextgrid'][j,:,:] * (1+VAT) + a_grid[np.newaxis, :]  - sol[h]['I'][j,:,:]
            rhs = R * a_grid         
            sol[h]['c'][j,:,:]  =     utils.interpolate_y(lhs,  rhs, sol[h]['c_nextgrid'][j,:,:])
      
        sol[h]['a'] = R * a_grid[np.newaxis, np.newaxis, :]   + sol[h]['I'] - sol[h]['c']  * (1+VAT) 
        
        # check borrowing constraint 
        sol[h]['a'], sol[h]['c'] = constr(sol[h]['a'], sol[h]['c'], a_grid[0], 'a', R, a_grid, VAT, sol[h]['I'])
        
   
    # unpack from dict and aggregate   
    cN  =  np.reshape(sol['N']['c'] , (ne*nBeta, nA))  
    cS  =  np.reshape(sol['S']['c'] , (ne*nBeta, nA)) 
    aN  =  np.reshape(sol['N']['a'] , (ne*nBeta, nA))  
    aS  =  np.reshape(sol['S']['a'] , (ne*nBeta, nA)) 

    
    dN = N_ / ssN
    dU = (1-N_) / (1-ssN)
    
    IncN  = np.reshape(sol['N']['I'], (ne*nBeta, nA))        
    IncS  = np.reshape(sol['S']['I'], (ne*nBeta, nA))        
    IncNpretax  = np.reshape(sol['N']['I']    + sol['N']['tInc']  * w * e_grid[np.newaxis, :, np.newaxis] , (ne*nBeta, nA))   
    IncSpretax  = np.reshape(sol['S']['I']    + sol['S']['tInc']  * U_inc[np.newaxis, :, np.newaxis] , (ne*nBeta, nA))   
     
    Ntaxes = sol['N']['tInc']  * w   * e_grid[np.newaxis, :, np.newaxis]
    Ntaxes =  np.reshape(np.broadcast_to(Ntaxes, (nBeta,ne, nA)), (ne*nBeta, nA))     
    Staxes = np.reshape(sol['S']['tInc']  * U_inc[np.newaxis, :, np.newaxis]  , (ne*nBeta, nA))   

    
    mu_N = cN ** (-1 / eis)
    mu_S = cS ** (-1 / eis)
    # EnVa =  R[np.newaxis, :] *  (destr * (1-Eq)     * mu_S   + (1 - destr * (1-Eq)) * mu_N)
    # EuVa =  R[np.newaxis, :] *  ((1-Eq* (1-destrO)) * mu_S   + Eq * (1-destrO)      * mu_N) 
    EnVa =  R[np.newaxis, :] *  (destr_L * (1-Eq)     * mu_S   + (1 - destr_L * (1-Eq)) * mu_N)
    EuVa =  R[np.newaxis, :] *  ((1-Eq* (1-destrO_L)) * mu_S   + Eq * (1-destrO_L)      * mu_N) 
                
    Incagg = N_ * IncN + (1-N_) * IncS

    # Aggregate 
    EVa =   np.reshape( np.stack((EnVa, EuVa), axis=-2), (ne*nBeta*nN, nA)) 
    
    # a =  np.reshape( np.stack((aN * dN, aS * dU), axis=-2), (ne*nBeta*nN, nA)) 
    # c = np.reshape( np.stack((cN* dN, cS* dU), axis=-2), (ne*nBeta*nN, nA)) 
    
    a =  np.reshape( np.stack((aN , aS ), axis=-2), (ne*nBeta*nN, nA)) 
    c = np.reshape( np.stack((cN, cS), axis=-2), (ne*nBeta*nN, nA)) 
    
    UincAgg = np.reshape(np.broadcast_to( U_inc[np.newaxis, :, np.newaxis], (nBeta,ne, nA)), (ne*nBeta, nA)) 
        

    zeromat = np.zeros([ne*nBeta, nA])
    

    tInc = np.reshape( np.stack((Ntaxes * dN, dU * Staxes), axis=-2), (ne*nBeta*nN, nA)) 
    UincAgg =np.reshape( np.stack((zeromat,  UincAgg), axis=-2), (ne*nBeta*nN, nA)) 
    
    Inc = np.reshape( np.stack((IncN * dN, IncS* dU), axis=-2), (ne*nBeta*nN, nA)) 
    IncN = np.reshape( np.stack((IncN , zeromat), axis=-2), (ne*nBeta*nN, nA)) 
    IncS = np.reshape( np.stack((zeromat, IncS), axis=-2), (ne*nBeta*nN, nA)) 
    
    #print(N)
    #print(dU)
    
    # ctd = np.reshape( np.stack((cN* N + cS* (1-N), cN* N + cS* (1-N) ), axis=-2), (ne*nBeta*nN, nA)) 
    # atd = np.reshape( np.stack((aN* N + aS* (1-N) , aN* N + aS* (1-N) ), axis=-2), (ne*nBeta*nN, nA)) 
    ctd = np.reshape( np.stack((cN, cS  ), axis=-2), (ne*nBeta*nN, nA)) 
    atd = np.reshape( np.stack((aN ,  aS), axis=-2), (ne*nBeta*nN, nA)) 
        
    cN = np.reshape( np.stack((cN , zeromat ), axis=-2), (ne*nBeta*nN, nA)) 
    cS = np.reshape( np.stack((zeromat , cS ), axis=-2), (ne*nBeta*nN, nA)) 
    aN = np.reshape( np.stack((aN , zeromat ), axis=-2), (ne*nBeta*nN, nA)) 
    aS = np.reshape( np.stack((zeromat , aS ), axis=-2), (ne*nBeta*nN, nA)) 

    taxN = np.reshape( np.stack((Ntaxes , zeromat ), axis=-2), (ne*nBeta*nN, nA))  
    taxS = np.reshape( np.stack((zeromat , Staxes ), axis=-2), (ne*nBeta*nN, nA)) 

    a_debt = np.reshape(np.broadcast_to( (1-x) * a_grid[np.newaxis, np.newaxis, :], (nBeta,ne, nA)), (ne*nBeta, nA)) 
    
    
    a_debtN = np.reshape( np.stack(( a_debt ,zeromat ), axis=-2), (ne*nBeta*nN, nA)) 
    a_debtS = np.reshape( np.stack((zeromat, a_debt ), axis=-2), (ne*nBeta*nN, nA)) 
    a_debt = np.reshape( np.stack((dN * a_debt ,dU * a_debt ), axis=-2), (ne*nBeta*nN, nA)) 
    
    return EVa, a, c, tInc, UincAgg, Inc, cN, cS, aN, aS, ctd, atd, taxN, taxS, a_debt, a_debtN, a_debtS, IncN, IncS



# apply constraint on lower bound 
def constr(a, c, lb, var, R, a_grid, VAT, I):
    """x is a numpy array, lower bound lb is a scalar"""
    m = a + c * (1+VAT)  
    a_min = lb 
    c_con = np.minimum( m - a_min, c * (1+VAT)) / (1+VAT)
    
    Wealth = R * a_grid
    m_new = I + Wealth[np.newaxis, np.newaxis, :]
    a_con = m_new - c_con * (1+VAT) 

    return a_con, c_con


@njit
def util(c, eis):
    """Return optimal c, n as function of u'(c) given parameters"""
    utility = c**(1-1/eis)/(1-1/eis) 
    return utility


@njit(fastmath=True)
def mu(c, eis):
    """Return optimal c, n as function of u'(c) given parameters"""
    return c ** (-1 / eis)


@njit(fastmath=True)
def cut(c, eis):
    """Return optimal c, n as function of u'(c) given parameters"""
    return c**(1-1/eis)/(1-1/eis)


@njit
def inv_mu(uc, eis):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis) 

 
def mTaxf(I, ssAvgInc, cs_marg):
    """Return marginal tax rate as function of income"""
    inc_pos = I / ssAvgInc # relative income 
    mTax = cs_marg(inc_pos)
    return mTax  

 
def avgTaxf(I, ssAvgInc, cs_avg):
    """Return avgerage tax rate as function of income"""
    inc_pos = I / ssAvgInc # relative income 
    avgTax = cs_avg(inc_pos)
    return avgTax  



def transfers(pi_e, Div, e_grid, dist):    
    ne = len(e_grid)
    e = np.arange(1,ne+1)    
    x = (e**dist  ) 
    div = Div *  x / np.sum(pi_e * x ) 
    return div
 
def transfers_td_e(pi_e, Div, e_grid):
    T = Div  * e_grid / np.sum(pi_e * e_grid)
    return T


def res_calib_2(x, *args):
    
    beta_mid, beta_var, beta_disp, vphi, kappa  = x
    
    args_dict = args[0]

    beta, pi_beta, Pi_beta = beta_dist(args_dict['nBeta'], beta_mid, beta_var, beta_disp, args_dict['dist_type'] )

    Pi   =  np.kron(np.kron(Pi_beta, args_dict['Pi_ern']), args_dict['Pi_N'] )
    pi_e =  np.kron(np.kron(pi_beta, args_dict['pi_ern']),args_dict['pi_N'])
    Pi_seed = pi_e 
    
    penalty_var = 0 
    if beta_var < 0.00001:
        beta_var = 0.00005
        penalty_var += 1 + abs(beta_var)
        
    if vphi < 0:
        vphi = 0.05

    penalty = 0 
    if max(beta) > args_dict['beta_max'] - 0.0005:
        beta_mid = args_dict['beta_max'] - beta_var - 0.0005
        penalty += abs(1+x[0]+x[1])

    if min(beta) < 0.2:
        beta_mid = 0.2 + beta_var + 0.0001
        penalty += abs(1+x[0]+x[1])

    beta, pi_beta, Pi_beta = beta_dist(args_dict['nBeta'], beta_mid, beta_var, beta_disp, args_dict['dist_type'] )
    Pi   =  np.kron(np.kron(Pi_beta, args_dict['Pi_ern']), args_dict['Pi_N'] )
    pi_e =  np.kron(np.kron(pi_beta, args_dict['pi_ern']),args_dict['pi_N'])
    Pi_seed = pi_e 

    print('betas', beta_mid, beta_var, beta_disp )
    print(max(beta))
    print('Labor', vphi )
    print('kappa', kappa )
    
    
    args_dict.update({  'beta' : beta, 'pi_beta' : pi_beta, 'Pi_beta' : Pi_beta, 'Pi' : Pi, 'pi_e' : pi_e, 'vphi' : vphi, 'kappa' : kappa})
                      

    out =   EGMhousehold.ss(**args_dict)        
        
    sBot, sMiddle, sTop, s10, sborrow_con, share_borrow = IneqStat(out, args_dict['nPoints'], args_dict['a_lb'])
    # top 10%: 50% of wealth
    # middle 40% : 45 %
    # bottom 50% : 5% 

    taxes = out['C'] - args_dict['Agg_C'] 
    
    # targets
    if args_dict['a_grid'][0] < 0:
        target_bot    = 0
        target_neg  = 0.15
    else:
        target_bot    = 0
        target_middle = 0.1

    objective_func =    abs(out['A'] - ( args_dict['p'] + args_dict['B']) + penalty)   +  abs(sBot- target_bot + penalty + penalty_var) +  abs(target_neg - share_borrow +   penalty_var) + abs(out['NS'] -1) 

    print('A-B', ((out['A'] - (args_dict['p'] + args_dict['B']))  ), 'Middle s', ((sMiddle - 0.45)), 'Bottom',((sBot- target_bot )), 'neg',target_neg - share_borrow , 'labor', out['NS'] -1)

    
    return objective_func


def res_calib_2_root(x, *args):
    
    beta_mid, beta_var, beta_disp  = x
 
    args_dict = args[0]

    beta, pi_beta, Pi_beta = beta_dist(args_dict['nBeta'], beta_mid, args_dict['beta_var'] , args_dict['beta_disp'] , args_dict['dist_type'] )
    Pi   =  np.kron(Pi_beta, args_dict['Pi_ern'])
    pi_e =  np.kron(pi_beta, args_dict['pi_ern'])    
    
    penalty_var = 0 
    if beta_var < 0.00001:
        beta_var = 0.00005
        beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
        Pi   =  np.kron(Pi_beta, Pi_ern)
        pi_e =  np.kron(pi_beta, pi_ern)   
        penalty_var += 1 + abs(beta_var)
    
    beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
    Pi   =  np.kron(Pi_beta, Pi_ern)
    pi_e =  np.kron(pi_beta, pi_ern)
     
 



    penalty = 0 
    if max(beta) > beta_max:
        beta_mid = beta_max - beta_var - 0.0001

        beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
        Pi   =  np.kron(Pi_beta, Pi_ern)
        pi_e =  np.kron(pi_beta, pi_ern)  
        penalty += abs(1+x[0]+x[1])


    if min(beta) < 0.2:
        beta_mid = 0.2 + beta_var + 0.0001

        beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
        Pi   =  np.kron(Pi_beta, Pi_ern)
        pi_e =  np.kron(pi_beta, pi_ern)  
        penalty += abs(1+x[0]+x[1])


    print(beta_mid, beta_var, beta_disp )

        
    out =   EGMhousehold.ss(EnVa = EnVa, EuVa = EuVa, Pi = Pi, Pi_ern = Pi_ern, a_grid = a_grid, Pi_seed =  pi_e, e_grid = e_grid, 
                     pi_e = pi_e, pi_ern = pi_ern, w = w  , ra = ra, beta = beta, eis = eis,
                     q = q, N = N, destr = destr, b = b, T_dist = T_dist, pi_beta = pi_beta, 
                           Tss = Tss, Ttd = 0, Tuni = 0, ssAvgInc = ssAvgInc, VAT = VAT,  
                           nPoints = nPoints,  cs_avg = cs_avg, ttd_inf = ttd_inf,  tdiv = tdiv, ssflag=True)  
      
        
    sBot, sMiddle, sTop, s10, sborrow_con = IneqStat(out, nPoints)

    print('A-B', ((out['A'] - (B+p))  ), 'Middle s', ((sMiddle - 0.45)), 'Bottom',((sBot- 0.05 )), 'Borrow',sborrow_con - 0.07  )

    return out['A'] - (B+p), sMiddle - 0.45, sborrow_con - 0.07

def res_calib_3(x, *args):
    
    beta_mid  = x
 
    (EnVa, EuVa, Pi, Pi_ern, a_grid, pi_e, 
                     e_grid, pi_ern, w, ra, eis,
                     q, N, destr, b, T_dist, 
                     div, lumpsum_T, ssAvgInc, VAT, pi_p,  
                     nPoints, cs_avg, nBeta, dist_type, 
                     beta_var, beta_disp, B, K, beta_max, phi_a, rho_a, ttd_inf, p, tdiv, Tss) = args 
    
    print(x )
    
    beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
    Pi   =  np.kron(Pi_beta, Pi_ern)
    pi_e =  np.kron(pi_beta, pi_ern)
    

    print(      max(beta)   )    
    
    penalty = 0 
    if max(beta) > beta_max -0.001:
        beta_mid = beta_max - beta_var - 0.001
        beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
        Pi   =  np.kron(Pi_beta, Pi_ern)
        pi_e =  np.kron(pi_beta, pi_ern)  
        penalty += abs(1+x) 
    
    
    
    
    out =   EGMhousehold.ss(EnVa = EnVa, EuVa = EuVa, Pi = Pi, Pi_ern = Pi_ern, a_grid = a_grid, Pi_seed =  pi_e, 
                             e_grid = e_grid, pi_e = pi_e, pi_ern = pi_ern, w = w  , ra = ra, beta = beta, eis = eis,
                             q = q, N = N, destr = destr, b = b, T_dist = T_dist, pi_beta = pi_beta, phi_a = phi_a, rho_a = rho_a,
                             Tss = Tss, Ttd = 0, Tuni = 0, ssAvgInc = ssAvgInc, VAT = VAT,  
                             nPoints = nPoints, cs_avg = cs_avg, ttd_inf = ttd_inf, tdiv = tdiv, ssflag=True)

    objective_func =   (out['A'] - (B+p))**2
    print('A-B', (out['A'] - (B+p)))
    return objective_func

def res_calib_3_root(x, *args):
    
    beta_mid = x
    print(x)
    args_dict = args[0]

    beta, pi_beta, Pi_beta = beta_dist(args_dict['nBeta'], beta_mid, args_dict['beta_var'] , args_dict['beta_disp'] , args_dict['dist_type'] )
    
    penalty = 0 
    if max(beta) > args_dict['beta_max'] -0.0001:
        print('PENALTY')
        beta_mid = args_dict['beta_max'] - args_dict['beta_var'] - 0.0001
 
        penalty += abs(1+x) 

    beta, pi_beta, Pi_beta = beta_dist(args_dict['nBeta'], beta_mid, args_dict['beta_var'] , args_dict['beta_disp'] , args_dict['dist_type'] )
    Pi   =  np.kron(np.kron(Pi_beta, args_dict['Pi_ern']), args_dict['Pi_N'] )
    pi_e =  np.kron(np.kron(pi_beta, args_dict['pi_ern']), args_dict['pi_N'])
    Pi_seed = pi_e
    
        
    args_dict.update({ 'beta' : beta, 'pi_beta' : pi_beta, 'Pi_beta' : Pi_beta, 'Pi' : Pi, 'pi_e' : pi_e, 'Pi_seed' : Pi_seed})

    out =   EGMhousehold.ss(**args_dict)

    Asset_mkt =   out['A'] - (args_dict['B'] + args_dict['p'])
    if args_dict['h_con'] :
        labor = 0 
    print('A-B', (Asset_mkt))
    return [Asset_mkt ]



#%%    


def SAM_calib(U, destr, settings):
    N = 1-U  
    if settings['SAM_model']  == 'Standard':
        S = 1 - (1-destr) * N
        q = N * destr / S      
        pMatch = 0.7  # UNEMPLOYMENT AND BUSINESS CYCLES 
        V = S * q / pMatch   
        Tight = V / S 
    
        def Match_Func_res(x):
            res = S * q -  S * V / ((S**x + V**x)**(1/x))
            return res
        ma = optimize.fsolve(Match_Func_res, [1.4])    
        destrNO = destr
        destrO = 0
        nPos = 1
    elif settings['SAM_model'] == 'Costly_vac_creation':
        if settings['SAM_model_variant']  == 'simple':
            destrNO =  destr
            destrO = 0 
            S = 1 - (1-destr) * N        
            q = (N * destr / S)    
            pMatch = 0.7
            V = S * q / pMatch   
            Tight = V / S 
            nPos = V * pMatch
            
            def Match_Func_res(x):
                res = S * q -  S * V / ((S**x + V**x)**(1/x))
                return res
            ma = optimize.fsolve(Match_Func_res, [1.4])            
                           
            
        elif settings['SAM_model_variant'] == 'FR': # Fujita & Ramey 
            destrO = 0.5 * destr
            destrNO = (destr-destrO)/ (1-destrO)
            S = 1 - (1-destr) * N        
            q = (N * destr / S)/(1-destrO)      
            pMatch = 0.7
            V = S * q / pMatch   
            Tight = V / S 
        
            def Match_Func_res(x):
                res = S * q -  S * V / ((S**x + V**x)**(1/x))
                return res
            ma = optimize.fsolve(Match_Func_res, [1.4])                
            rhs = (1-destrO) * V + (1-destrO) * destrNO * N -  (1-destrO) * pMatch * V 
            nPos = V - rhs             
        
    return S, q, pMatch, V, Tight, ma, destrO, destrNO, nPos 
    




def ss_calib(calib, settings): 
    # Load income tax functions 
    with open('Tax_Function_Fit/cs_avg.pickle', 'rb') as f:
        cs_avg = pickle.load(f)
    with open('Tax_Function_Fit/cs_marg.pickle', 'rb') as f:
        cs_marg = pickle.load(f)


    # Grids
    nA = 250
    nBeta = 5  
    ne = 11 
    nN = 2
    amax = 50  
   
    e_grid, pi_ern, Pi_ern = utils.markov_rouwenhorst(rho= 0.94, sigma= 0.7, N=ne) # Values from IMPC: rho = 0.91, variance = 0.92. Floden and linde (2001) for Sweden persistence values. 

    nPoints = [nBeta, ne, nA]
 
    beta_mid_guess  = 0.96513206
    beta_var_guess  =  0.02
    beta_disp_guess =  0.06
    vphi_guess       = 2.125509040803471
    nonlintax_guess  =  0.049834087883106254

    kappa_g = 0.04
    

    dist_type = "LeftSkewed"
    beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid_guess, beta_var_guess, beta_disp_guess, dist_type )
    

    N = 0.95  
    b_ratio = 0.51

    mix = 0
    destr = 0.1
    eis = 0.5 
    frisch = 0.5 
    alpha = 0.35

    G = 0.24
    VAT = 0
    U = 0.05
          
    Y = 1 
    N = 1 - U   

    S, q, pMatch, V, Tight, ma, destrO, destrNO, nPos  = SAM_calib(U, destr, settings)
    Eq = q 
    
    pi_N = [N,1-N]    
    Pi_N = np.empty([2,2])
    Pi_N[0,0] = 1 - destr * (1-q)
    Pi_N[0,1] = destr * (1-q)
    Pi_N[1,1] = 1 - q * (1-destrO)
    Pi_N[1,0] = q * (1-destrO)
   
    Pi   =  np.kron(np.kron(Pi_beta, Pi_ern), Pi_N)
    pi_e =  np.kron(np.kron(pi_beta, pi_ern), pi_N)
    Pi_seed = pi_e
    
    r = 0.02 / 4 # 2% yearly    
    rstar = r   
    P = 1
    pi = 0
    i = r + pi 
   

    Benefit_type = 0 # proportional to e_grid  
    vacK_rate = 0.05
    w_g = 0.58 
    vacK_g = vacK_rate * w_g * N

    K = 8 
    mup = 1.1
    mc = 1 / mup 
    
    destr_L = destr
    destrO_L = destrO 
    
    
    A_tot = 2.3
    B  = A_tot * 0.5    
    p = A_tot - B
    T_firms_g = 0 
    F_cost_g = 0.05
    Agg_C_guess  = 0.4
    
    
    if settings['Fvac_share']:
        Fvac_factor = settings['Fvac_factor']
    else: 
        Fvac_factor = 5 

    def Asset_calib(x):
        T_firms_g, Agg_C_g, vacK_g, F_cost_g, w_g = x

        U_inc_g, U_inc_agg_g  = Unemp_benefit(Benefit_type, w_g * b_ratio, e_grid, pi_ern, ne, w_g)
        ssAvgInc =  N * w_g   + (1-N) * U_inc_agg_g
        tax_N  = N            * np.vdot(avgTaxf(w_g   * e_grid, ssAvgInc, cs_avg) * w_g * e_grid , pi_ern)
        tax_S  = (1-N)        * np.vdot(avgTaxf(U_inc_g, ssAvgInc, cs_avg) * U_inc_g, pi_ern)    
        
        
        mc = 1/mup
        rk = alpha * mc * Y / K
        delta = rk - r 
        I = K * delta  
        
        b = optimize.fsolve(b_calib, [w_g * 0.5], args = (Benefit_type, e_grid, pi_ern, ne, w_g, b_ratio, N, cs_avg, avgTaxf) )

        U_inc, U_inc_agg  = Unemp_benefit(Benefit_type, b, e_grid, pi_ern, ne, w_g)               
        MPL = (1-alpha) * mc * Y / N         

        if settings['SAM_model'] == 'Standard':
            JV = 0
            JM = (MPL - w_g) / (1 -  (1-destr) /(1+r))
            LS_res = JV - (- vacK_g + pMatch * JM)
            Vac_costs = vacK_g  
        elif settings['SAM_model'] == 'Costly_vac_creation':            
            if Fvac_factor == np.inf:
                Fvac  = vacK_g / nPos 
                vacK_g = 0 
            else:
                if settings['Fvac_share']:
                    Tot_cost = vacK_g #+ Fvac * nPos
                    Fvac  = Fvac_factor * Tot_cost / nPos
                    vacK_g =  Tot_cost - Fvac * nPos
                else:
                    Fvac  = Fvac_factor / (nPos / vacK_g)
                JV = Fvac * nPos
                
            if settings['SAM_model_variant']  == 'simple':
                JM = (MPL - w_g) / (1 -  (1-destr) /(1+r))
                LS_res = JV - (- vacK_g + pMatch * JM + (1-pMatch) * JV / (1+r))    
            else:              
                 JM  = (JV - (1-pMatch) *(1-destrO) * JV /(1+r) + vacK_g) / pMatch
                 LS_res = JM - ((MPL-w_g) + (1-destrO) * (1-destrNO)/(1+r) * JM  + destrNO * (1-destrO)  * JV /(1+r))                                            
                
            Vac_costs = vacK_g  + Fvac * nPos**2
            
        div = 1 - w_g * N  - I - Vac_costs - F_cost_g - T_firms_g
        p_res = div / r - p
       
        A_tot_ = p + B
        G_rev = tax_N + tax_S +  VAT * Agg_C_g   +  B + T_firms_g 
        G_exp = G +  U_inc_agg * (1-N) +  B * (1 + r)  
            
        lumpsum_T_res = G_rev - G_exp
        Agg_C  = (ssAvgInc - tax_N - tax_S   + A_tot_ * r ) / (1+VAT) 

        A2I =  (p  +   B) / (ssAvgInc - tax_N - tax_S + A_tot_ * r - A_tot_ * kappa_g * 0.15 )  

        w_res = ( MPL / w_g-1)   -  settings['vac2w_costs']

        return  np.array([lumpsum_T_res, Agg_C - Agg_C_g, LS_res, p_res, w_res]) 
    

    sol = optimize.root(Asset_calib, np.array([T_firms_g, Agg_C_guess, vacK_g, F_cost_g, w_g]),  method='hybr')
    (T_firms, Agg_C, vacK, F_cost, w) = sol.x 
    if not sol.success:
        raise Exception("Solver did not succeed") 
    lumpsum_T = 0 
    rk = alpha * mc / K
    delta = rk - r 
    I = K * delta  
    Fvac  = Fvac_factor / (nPos / vacK)
    
    assert w > 0 
    assert vacK >= 0 

    wss = w 
    sBorrow = 0.5
    a_lb = -w  * sBorrow
    a_grid = nonlinspace(a_lb , amax, nA, 1.5)         
    
    b = optimize.fsolve(b_calib, [w * 0.5], args = (Benefit_type, e_grid, pi_ern, ne, w, b_ratio, N, cs_avg, avgTaxf))
    U_inc, U_inc_agg  = Unemp_benefit(Benefit_type, b, e_grid, pi_ern, ne, w) 
    ssAvgInc =  N * w   + (1-N) * U_inc_agg # average taxable labor income in steady state     
    tax_N  = N            * np.vdot(avgTaxf(w   * e_grid, ssAvgInc, cs_avg) * w * e_grid , pi_ern)
    tax_S  = (1-N)        * np.vdot(avgTaxf(U_inc, ssAvgInc, cs_avg) * U_inc, pi_ern)
    
           
    Z = Y / (K ** alpha * N**(1-alpha))
    Zss = Z     

    MPL = (1-alpha) * mc * Y / N


    Tight = V / S 
    if settings['SAM_model'] == 'Standard':
        Vac_costs = vacK    
        JV = 0
        JM = (MPL - w) / (1 -  (1-destr) /(1+r))            
    elif settings['SAM_model'] == 'Costly_vac_creation':
        if settings['Fvac_share']:
                Tot_cost = vacK #+ Fvac * nPos
                Fvac  = Fvac_factor * Tot_cost / nPos
                vacK =  Tot_cost - Fvac * nPos            
        else:
            if Fvac_factor == np.inf:
                Fvac  = vacK / nPos 
                vacK = 0 
            else:
                Fvac  = Fvac_factor / (nPos / vacK)            
        JV = Fvac * nPos            
        Vac_costs = vacK  + Fvac * nPos**2
        if settings['SAM_model_variant']  == 'simple':
            JM = (MPL - w) / (1 -  (1-destr) /(1+r))  
        else:
            JM  = (JV - (1-pMatch) *(1-destrO) * JV /(1+r) + vacK) / pMatch
        
    div =  1 - w * N  - I - Vac_costs - F_cost - T_firms

    p = div / r
    
    MF_Div = 0
    mix = 0
    ra_test =  (div + p)  / A_tot  + (1+r ) * B / A_tot - (1  + r)   
    ra = r
    
    assert abs(ra_test) < 1e-07
    
    Agg_C  = (ssAvgInc  - tax_N  - tax_S   + lumpsum_T + A_tot * r ) / (1+VAT) 

    walras1 = 1 - Agg_C  - I - G - Vac_costs  - F_cost
    print('Walras 1', walras1)
    #assert abs(walras1) < 1e-8
    #print((100*Vac_costs/0.7)/(w))

    beta_max = 1/(1+ra)     
    assert  max(beta)  < beta_max
    
    Tss = lumpsum_T + nonlintax_guess
    T_dist = 0.1
    T = transfers(pi_ern, Tss, e_grid, T_dist)     
    bnds = [-8, 8]
    args = (U_inc, w, pi_ern, e_grid, N, div , Tss, ssAvgInc, cs_avg, transfers, avgTaxf)
    res = optimize.minimize_scalar(Income_gini, np.array([T_dist]), method='Bounded',  bounds = bnds, args=args)  
    T_dist = res.x 
    Income_gini_disp(x = T_dist, args = args)
    #Tss = lumpsum_T
    Ttd = 0
    ttd_inf = {'p75' : 0, 'ss_dist' : np.zeros((ne*nBeta, nA)) }


    # initialize guess for policy function iteration
    s_match = True     
    use_saved = settings['use_saved']
    save = settings['save']
    if use_saved == True:
        try:
            npzfile = np.load('init_values.npz')        
            EVa = npzfile['x']
            if (EVa.shape == (nBeta*ne*nN, nA)) is False: # check that loaded values match in shape 
                s_match = False            
        except OSError as e:
            s_match = False

    
    if s_match == False or use_saved == False:        
        coh_n  =  np.reshape( (1-avgTaxf(w   * e_grid , ssAvgInc, cs_avg)) * w * e_grid + T,   ((1, ne, 1))) 
        coh_s  =  np.reshape( (1-avgTaxf(b  * e_grid, ssAvgInc, cs_avg))   * b * e_grid + T ,   ((1, ne, 1)))  
        coh_n  = np.reshape(np.broadcast_to(coh_n, (nBeta,ne, nA)), (ne*nBeta, nA))
        coh_s  = np.reshape(np.broadcast_to(coh_s, (nBeta,ne, nA)), (ne*nBeta, nA))    
        EnVa =  (1 + r) * (0.8  * coh_n) ** (-1 / eis) 
        EuVa =  (1 + r) * (0.8  * (coh_s )) ** (-1 / eis) 
        EVa = np.reshape(np.stack((np.reshape(EnVa, (nBeta, ne, nA)), np.reshape(EuVa, (nBeta, ne, nA))), axis=-2), (nBeta* ne*nN, nA))
        
    ssc = np.ones([EVa.size])
    #beta_mid_guess, beta_var_guess, beta_disp_guess, obj  = init_search()
    lumpsum_T_init = lumpsum_T
    Tss = lumpsum_T 

    args = {}
    args = { 'EVa' : EVa,  'Pi' : Pi, 'dist_type' : dist_type, 'nBeta' : nBeta, 'Pi_ern' : Pi_ern, 'a_grid' : a_grid, 'pi_e' : pi_e, 'ssN' : N, 
             'e_grid' : e_grid, 'pi_ern' : pi_ern, 'w' : w, 'ra' : ra, 'eis' : eis, 'Eq' : q, 'N' : N, 'destr_L' : destr, 'U_inc' : U_inc, 'T_dist' : T_dist,
             'div' : div, 'lumpsum_T_init' : lumpsum_T_init, 'ssAvgInc' : ssAvgInc, 'VAT' : VAT, 'pi' : pi, 'rstar' : rstar, 'MF_Div' : MF_Div, 'mix' : mix, 'wss' : w, 
             'nPoints' : nPoints, 'cs_avg' : cs_avg, 'dist_type' : dist_type, 'B' : B, 'rss' : r, 'p' : p , 'sBorrow' : sBorrow, 'hss' : 1,  'T_firms' : T_firms,
             'beta_max' : beta_max, 'ttd_inf' : ttd_inf, 'K' : K, 'Tss' : Tss, 'frisch' : frisch, 'G' : G, 'P' : P, 'P_lag' : P, 'N_' : N, 
             'tax_N' : tax_N, 'a_lb' : a_lb, 'b' : b, 'Benefit_type' : Benefit_type, 'Pi_seed' : Pi_seed, 'Ttd' : Ttd, 'Tuni' :0, 'pi_N' : pi_N, 'Pi_N' : Pi_N,
             'Agg_C' : Agg_C, 'tax_S' : tax_S, 'destrO_L' : destrO, 'ssflag' : True}      
    
    
    if (calib == 'Full_calib') or (calib == 'Partial_calib'):           
            if calib == 'Full_calib':
                
                res1 = optimize.minimize(res_calib_2, np.array([beta_mid_guess, beta_var_guess, beta_disp_guess, vphi_guess, kappa_g]), method='Nelder-Mead',   args=args, tol = 1e-5)  
                
                beta_mid_guess, beta_var, beta_disp, vphi_guess, nonlintax_guess, kappa  = res1.x               
                beta_var_guess  = beta_var   
                beta_disp_guess = beta_disp
                kappa_g = kappa
            
            # partial calib    
            kappa = kappa_g
            beta_var  = beta_var_guess   
            beta_disp = beta_disp_guess    
            args.update({'beta_var' : beta_var, 'beta_disp' : beta_disp, 'kappa' : kappa})     
                        
                   
            print('Beta distribution:', beta_mid_guess, beta_var_guess, beta_disp_guess ) 
            print('Labor supply', vphi_guess, nonlintax_guess)
            

            res = optimize.root(res_calib_3_root, [beta_mid_guess], args=args, method='hybr')
            
            beta_mid = res.x          
            beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid, beta_var, beta_disp, dist_type )
            Pi   =  np.kron(np.kron(Pi_beta, Pi_ern), Pi_N)
            pi_e =  np.kron(np.kron(pi_beta, pi_ern), pi_N)        
            Pi_seed = pi_e    
    elif calib == 'Solve' : 
            beta_mid  = beta_mid_guess
            beta_var  = beta_var_guess
            beta_disp = beta_disp_guess  
            vphi      = vphi_guess
            kappa = kappa_g                                
    else : 
            raise ValueError('No calibration chosen!')

    args.update({'beta' : beta, 'kappa' : kappa, 'Pi_seed' : Pi_seed, 'pi_e' : pi_e, 'Pi' : Pi, 'pi_beta' : pi_beta })   
    print(beta_mid, beta_var, beta_disp)
    ss =   EGMhousehold.ss(**args)    
    if save == True:
        np.savez('init_values', x = ss['EVa'])

    # Short run parameters not identified/needed in SS 
    phi = 1.3
    kappak = 6
    eps_p = mup / (mup-1)
    kappap =  (eps_p-1) / 0.03 

    Agg_C  = (ssAvgInc - tax_N - tax_S  + Tss + (p + B) * r) / (1+VAT) 


    sc_neg = np.nonzero(ss['a'] < 0)
    walras = 1 - ss['C']  - I - G - Vac_costs - F_cost + kappa * ss['A_DEBT']  
    G_rev = ss['TINC'] +  VAT * ss['C']   + B + T_firms 
    G_exp = G + ss['UINCAGG'] + Tss + B * (1 + r)   

    
    print('Walras 2', walras)   
    print('Asset market err', ss['A']-(B+p), 'G_budget', G_rev - G_exp)    
    
    ss.update({'goods_mkt' : walras})

    a_pop = ss['a']  
    sc = np.nonzero(ss['a'] < a_lb + 1e-7)
    print( 100 * sum(ss['D'][sc]))
    
    print( 100 * sum(ss['D'][sc_neg]))
    print(IneqStat(ss, nPoints, a_lb))
 
    ss.update({'V': V, 'vacK' : vacK, 'pMatch' : pMatch, 'q': q, 'N' : N, 'B': B, 'kappap': kappap, 'Y': Y, 'rstar': r, 'Z': Z, 'mup': mup, 'pi': pi, 'Pi' : Pi, 'eps_p' : eps_p,  'Pi_seed' : pi_e, 'MPL' : MPL, 'eps_r' : 0, 'mu' : 0, 'destr_L' : destr, 'destrO_L' : destrO,
               'K': K, 'alpha': alpha, 'delta': delta, 'I': I, 'S': S, 'G' : G,  'div' : div, 'phi' : phi, 'T_dist' : T_dist, 'Tuni' : 0, 'ttd_inf' : ttd_inf, 'L' :  N, 'U_inc' : U_inc, 'Agg_C' : Agg_C, 'p' : p, 'T_rate' : 1, 'ssN' : N, 'pshare' : p/ss['A'], 'destrO' : destrO, 'destrNO' : destrNO,
               'kappap': kappap, 'eis': eis, 'beta': beta,  'destr': destr, 'Q': 1,  'mc': mc,  'lumpsum_T' : lumpsum_T, 'Zss' : Zss, 'ra' : r, 'rk' : rk, 'r' : r, 'i' : i, 'Tightss' : Tight, 'ma' : ma, 'piw' : 0, 'a_lb' : a_lb, 'mix' : mix,'Pi_N' : Pi_N, 'pi_N' : pi_N,
               'Nss' : N, 'wss' : w,  'MF_Div' : MF_Div, 'G_rev' : G_rev, 'G_exp': G_exp, 'P' : 1, 'Tss' : lumpsum_T, 'Ttd' : Ttd, 'Z_I' : 1, 'Isip' :0, 'rss' : r, 'F_cost' : F_cost, 're' : r, 'b_ratio' : b_ratio, 'rg' : r, 'dist_type' : dist_type, 'Fvac' : Fvac,
               'VAT': VAT, 'pi_ern' : pi_ern, 'ssAvgInc' : ssAvgInc, 'b' : b,  'Tight' : Tight, 'psip' : 0, 'isip' : 0,  'K_cost' : 0, 'kappak' : kappak, 'cs_avg' : cs_avg, 'Bss' : B, 'epsI' :4, 'Benefit_type' : Benefit_type, 'div_' : div, 'UINCAGG_count' : ss['UINCAGG'], 'T_firms' : T_firms,
               'beta_mid' : beta_mid, 'beta_var' : beta_var, 'beta_disp' : beta_disp, 'uT' : 0,  'div_MF' : 0, 'nPos' : nPos, 'JV' : JV, 'JM' : JM, 'ssflag': False})    
     

    print('Average Beta' , np.vdot(beta, pi_beta))

    
    # Check bargaining set 
    upper_lvl =  (1-alpha) * mc / N
    lower_lvl = b

    assert w < upper_lvl
    assert w > lower_lvl  

    ss.update({'A_agg' : ss['A'], 'C_agg' : ss['C'], 'taxes' : ss['TINC']})
    
    # steady state values for endo. destr. rate 
    ss.update({'destrNOss' : destrNO, 'JMss' : JM, 'eps_m' : 0.5})
    ss.update({'destrOss' : destrO, 'JVss' : JV, 'eps_V' : 0.5, 'muV' : 0 })

    # Check that income is strictly positive 
    assert min(ss['Inc'].flatten()) > 0

    return ss 

    
#calib = 'Full_calib'     # Calibrate to all moments in wealth distribution
#calib = 'Partial_calib'   # Calibrate asset market equilibrium A = p + B
# calib = 'Solve'          # Solve with pre-specified values 
   

# settings = {'save' : True, 'use_saved' : True}
# settings['Fvac_share'] = False 

# # choose labor market 
# settings['SAM_model'] = 'Standard'
# #settings['SAM_model'] = 'Costly_vac_creation'
# #settings['SAM_model_variant'] = 'FR'
# settings['SAM_model_variant'] = 'simple'

# settings['endo_destrNO'] = False
# settings['endo_destrO'] = False 

# # some calibration for labor market 
# settings['Fvac_factor'] = 5
# settings['vac2w_costs'] = 0.05 

# # solve the steady state 
# ss = ss_calib(calib, settings)     


#%%   Impulses 
    

"(mc, Y) -> (nkpc)"
@simple
def pricing(pi, mc, Y, kappap, mup, r, rstar, eps_p):    
    eps_p = mup / (mup-1)
    nkpc = (1-eps_p) + eps_p * mc - kappap * (pi+1) * pi   + kappap * Y(+1) / Y *  (pi(+1)+1) * pi(+1)  / (1 + r(+1))
    return nkpc  


@solved(unknowns=['K', 'Q', 'I'], targets=['inv', 'val', 'K_res'])
def firm_investment(K,  alpha, rstar, delta, kappak, Q, mc, Y, Z_I, I, r):       
    rk_plus =  alpha * mc(+1) * Y(+1) / K 
    inv = Q - (rk_plus + Q(+1) * (1-delta))/(1+r(+1))
    LHS = 1 + kappak/2 * (I/I(-1) -1)**2   + I/I(-1) * kappak * (I/I(-1) -1) 
    RHS = Q(+1) * Z_I(+1)  + kappak * (I(+1)/I -1) * (I(+1)/I)**2   
    val = LHS - RHS     
    K_res = K - ((1 - delta) * K(-1) + I(-1) * Z_I) 
    return inv, val, K_res


# @solved(unknowns=['K', 'Q', 'I'], targets=['inv', 'val', 'K_res'])
# def firm_investment1(K,  alpha, rstar, delta, kappak, Q, mc, Y, Z_I, I, r):
#     epsI = 3 
#     MPK = alpha * mc(+1) * Y(+1) / K 
#     inv = (K/K(-1) - 1) / (delta * epsI) + 1 - Q * Z_I
    
#     val = MPK  - (K(+1)/K -
#             (1-delta) + (K(+1)/K - 1)**2 / (2*delta*epsI)) + K(+1)/K*Q(+1) - (1 + r(+1))*Q
#     K_res = K - ((1 - delta) * K(-1) + I * Z_I) 
#     return inv, val, K_res, MPK

@simple
def firm_labor_standard(mc, Y, N, alpha, pMatch, vacK, destr, w,  L, rstar, r, Z, JV, JM, mu, K):   
    MPL = (1-alpha)  * mc * Y / N   
    free_entry = JV - 0
    JV_res = JV - (- vacK + pMatch * JM)
    JM_res = JM - ((MPL - w - mu) + JM(+1) * (1-destr)/(1+r(+1)))         
    return  free_entry, JV_res, JM_res , MPL

@simple
def ProdFunc(Y, Z, K, alpha, L): 
    ProdFunc_Res = Y - Z * K(-1)**alpha * L**(1-alpha)
    return ProdFunc_Res
    

@solved(unknowns=['w'], targets=['w_res'])
def wages(Tight, w, wss, Tightss, pi): 
    eta =  0.005
    w_res = np.log(w) - (np.log(wss) + eta * np.log(Tight/Tightss) )    
    return w_res


@simple
def laborMarket1(q, N, destr, S, pMatch, Tight):
    N_res = N - ((1-destr(-1)) * N(-1) + S * q)    
    S_res = S - (1 - (1-destr(-1)) * N(-1))
    V = q * S / pMatch
    N_ = N
    return N_res, S_res, V, N_

@simple
def laborMarket2(Tight, ma): 
    q      = Tight / ((1+Tight**ma)**(1/ma))
    pMatch = q / Tight
    return q, pMatch


@simple 
def MutFund(B, A_agg, r, ra, div,  p, pshare):
    MF_Div = pshare * (div + p) / p(-1) + (1-pshare) * (1 + r) 
    MF_Div_res = 1+ra - MF_Div
    return  MF_Div_res, MF_Div

@solved(unknowns=['p'], targets=['equity'])
def arbitrage(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def dividend_standard(Y, w, N, pi, mup, vacK, V, I, kappap, F_cost, kappak, T_firms):
    psip = kappap  * pi ** 2 * Y / 2 
    Isip = kappak * (I/I(-1) -1)**2 * I /2
    div = Y  - w * N -   vacK  - I  - psip - Isip - F_cost - T_firms 
    return psip, Isip, div


@simple
def dividend_costly_vac(Y, w, N, pi, mup, vacK, V, I, kappap, F_cost, kappak, T_firms, nPos, Fvac, mu):
    psip = kappap  * pi ** 2 * Y / 2 
    Isip = kappak * (I/I(-1) -1)**2 * I /2
    div = Y - w * N  -  vacK - Fvac * nPos**2 - mu - I  - psip - Isip - F_cost - T_firms 
    return psip, Isip, div

@solved(unknowns=[ 'i',  'r'], targets=['i_res',  'fisher'])
def monetary(rstar, pi, i, r, phi):
    rho   = 0.8
    i_res = i - (rho * i(-1) + (1-rho) * (rstar + phi * pi))    
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return i_res, fisher 


@simple 
def fiscal_rev(C, VAT, B, taxes,  MF_Div, T_firms):  
    G_rev = taxes  + B  + T_firms 
    return G_rev

@simple 
def fiscal_exp(b, r, G, B, N, lumpsum_T, uT, UINCAGG): 
    G_exp = G + UINCAGG  + (lumpsum_T + uT)  + B(-1) * (1+r) 
    return G_exp

@simple 
def B_res(G_rev, G_exp):  
    B_res = G_rev - G_exp
    return B_res

@simple 
def HHTransfer(uT):  
    Tuni  = uT
    return Tuni

@simple 
def Fiscal_stab_G(B, Bss, lumpsum_T, P, rg):  
    rho = 0.8
    G = ss['G'] - 0.1 *  np.log(B/Bss)
    return  G

@simple
def Fiscal_stab_T(B, Bss):  
    uT =  - 0.2 *  np.log(B(-1)/Bss)
    Tuni = uT
    return  uT, Tuni


@simple
def Asset_mkt_clearing(A_agg, B, p):  
    Asset_mkt = B + p - A_agg 
    return Asset_mkt    

@simple
def Labor_mkt_clearing(N, L):     
    Labor_mkt = L  -  N
    return Labor_mkt    


@simple
def aggregate(CN, CS, AN, AS, N, ssN, TAXN, TAXS, UINCAGG, TINC, CTD, ATD):
    dN =  N/ssN
    dU = (1-N)/(1-ssN)  
    C_agg =  CTD  
    A_agg = ATD     
    taxes = TINC
    UINCAGG_count  =  UINCAGG * dU
    return C_agg, A_agg, taxes, UINCAGG_count



'''Simple FR LM'''
@simple
def firm_labor_costly_vac(N, mc, Y, alpha, pMatch, vacK, destr, w,  L, rstar, r, nPos, Fvac, JV, JM, mu):   
    MPL = (1-alpha) *  mc * Y / N
    free_entry = JV -  nPos * Fvac
    JV_res = JV - (- vacK + pMatch * JM + (1-pMatch) * JV(+1) /(1+r(+1)))
    JM_res = JM - ((MPL-w - mu) + JM(+1) * (1-destr)/(1+r(+1)))    
    return  free_entry, JV_res, JM_res, MPL

@simple
def laborMarket1_costly_vac(q, N, destr, S, pMatch, V):
    N_res = N - ((1-destr(-1)) * N(-1) + S * q )    
    S_res = S - (1 - (1-destr(-1)) * N(-1))
    Match_res = V * pMatch - q * S 
    nPos = V - (1-pMatch(-1)) * V(-1) 
    N_ = N
    return N_res, S_res, Match_res, N_, nPos 

'''Fujita & Ramey LM'''
@simple
def firm_labor_costly_vac_FR(N, mc, Y, alpha, pMatch, vacK, destr,destrO, w,  L, rstar, r, nPos, Fvac, JV, JM, mu, muV, destrNO):   
    MPL = (1-alpha) *  mc * Y / N
    free_entry = JV - nPos * Fvac
    JV_res = JV - (- vacK - muV + pMatch * JM +(1-pMatch) * (1-destrO) * JV(+1) /(1+r(+1)))
    JM_res = JM - ((MPL - w - mu) + (1-destrO) * (1-destrNO)/(1+r(+1)) * JM(+1) + destrNO * (1-destrO)  * JV(+1) /(1+r(+1)) )  
    return  free_entry, JV_res, JM_res, MPL

@simple
def laborMarket1_costly_vac_FR(q, N, destr, S, pMatch, destrO, destrNO, V):
    N_res = N - ((1-destr(-1)) * N(-1) + S * q * (1-destrO(-1)))    
    S_res = S - (1 - (1-destr(-1)) * N(-1))
    Match_res = V * pMatch - q * S 
    nPos = V - (1-destrO(-1)) * (V(-1) + destrNO(-1) * N(-1) -  V(-1) * pMatch(-1)) 
    N_ = N 
    return N_res, S_res, Match_res, nPos, N_ 

@simple
def destr_rate(destrO, destrNO):
    destr = destrNO * (1-destrO) + destrO
    return destr

@simple
def destr_rate_lag(destrO, destr):
    destr_L  =  destr(-1)
    destrO_L =  destrO(-1)
    return destr_L, destrO_L


@simple
def Eq(q):
    Eq  =  q
    return Eq

@simple
def endo_destr(destrNO, destrNOss, JMss, eps_m, rstar, JM, r, destrO):
    destrNO_Res = destrNO - destrNOss * ( (1+rstar) / (1+r(+1)) * JM(+1)/JMss)**(-eps_m)
    mu = destrNOss * eps_m / (eps_m-1) * JMss * (1-destrO) / (1+rstar) * (1 - ((1+rstar) / (1+r(+1)) *JM(+1)/JMss)**(1-eps_m))
    return mu, destrNO_Res 

@simple
def endo_destrO(destrO, destrOss, JVss, eps_V, rstar, JV, r):
    destrO_Res = destrO - destrOss * ( (1+rstar) / (1+r(+1)) * JV(+1)/JVss)**(-eps_V)
    muV = destrOss * eps_V / (eps_V-1) * JVss  / (1+rstar) * (1 - ((1+rstar) / (1+r(+1)) *JV(+1)/JVss)**(1-eps_V))
    return muV, destrO_Res 



# function used to to choose labor market model 
def Choose_LM(settings):
    if settings['SAM_model'] == 'Standard':
        dividend = dividend_standard
        LM = solved(block_list=[laborMarket1, laborMarket2, firm_labor_standard, wages, destr_rate],  
                    unknowns=['N', 'S', 'Tight',  'JV', 'JM'],
                    targets=[ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res'] )  
        if settings['endo_destrNO'] == True:
            LM = solved(block_list=[laborMarket1, laborMarket2, firm_labor_standard, wages, destr_rate, endo_destr],  
                        unknowns=['N', 'S', 'Tight',  'JV', 'JM', 'destrNO'],
                        targets=[ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )              
            
    elif settings['SAM_model'] == 'Costly_vac_creation':
        dividend = dividend_costly_vac
        if settings['SAM_model_variant'] == 'simple':    
            LM = solved(block_list=[laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wages, destr_rate],  
                        unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM'],
                        targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res'] )  
            if settings['endo_destrNO'] == True:
                LM = solved(block_list=[laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wages, destr_rate, endo_destr],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrNO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )                  
        elif settings['SAM_model_variant'] == 'FR': 
            LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wages, destr_rate],  
                        unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM'],
                        targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res'] )  
            if settings['endo_destrNO'] == True:
                LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wages, destr_rate, endo_destr],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrNO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )  
            elif settings['endo_destrO'] == True:
                LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wages, destr_rate, endo_destrO],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrO_Res'] )        
    return dividend, LM 
        
#dividend, LM = Choose_LM(settings)

# Asset_block_G = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, dividend, EGMhousehold,  arbitrage, MutFund, Fiscal_stab_G,  aggregate],
#                 unknowns=[ 'B',  'ra'],
#                 targets=[  'B_res','MF_Div_res'] )  

# Asset_block_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res,  arbitrage, EGMhousehold, MutFund, Fiscal_stab_T, aggregate],
#                 unknowns=[ 'B',  'ra'],
#                 targets=[  'B_res','MF_Div_res'] )  

# Asset_block_B = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, dividend, EGMhousehold,  arbitrage, MutFund,  aggregate],
#                 unknowns=[ 'B',  'ra'],
#                 targets=[  'B_res','MF_Div_res'] )  

# Asset_block_only_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, EGMhousehold,  arbitrage, MutFund,  aggregate, HHTransfer],
#                 unknowns=[ 'uT',  'ra'],
#                 targets=[  'B_res','MF_Div_res'] )  

# Asset_block_no_G = solved(block_list=[dividend, EGMhousehold,  arbitrage, MutFund,  aggregate],
#                 unknowns=[   'ra'],
#                 targets=[  'MF_Div_res'] )  

# prod_stuff = solved(block_list=[monetary, pricing, ProdFunc, firm_investment],
#                 unknowns=[ 'pi', 'Y'],
#                 targets=[  'nkpc' , 'ProdFunc_Res' ] )  

 
#Time = 300   

# markup shock 
# Ivar = 'mup'
# exogenous = [Ivar]  
# rhos = 0.6
# dZ =  0.13 *ss[Ivar] * rhos**(np.arange(Time))


# exogenous = [Ivar]   
# unknowns = ['L', 'mc']
# targets = ['Asset_mkt', 'Labor_mkt']
#block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, destr_rate_lag, dividend, Eq] 
 
# G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=False)
# dMat = FigUtils.Shock_mult(G_jac, dZ, Ivar) # multiply Jacobian and shock 

# tplot = 30 
# fig = FigUtils.FigPlot3x3_new(ss, dMat, Ivar, tplot, dZ)
# plt.show() 

 

