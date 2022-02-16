# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:07:36 2020

@author: Nicolai
"""


import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numba import vectorize, njit, jit, prange, guvectorize 
#from numba import float64 as nbfloat64
 

import utils
from het_block import het
from simple_block import simple
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import ticker

plt.style.use('ggplot')



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
#import hank
import determinacy as det

from consav.grids import nonlinspace
from statsmodels.nonparametric.kde import KDEUnivariate # weighted kernel density 

from types import SimpleNamespace
import FigUtils  

from quantecon import lorenz_curve

def beta_draw(mean, width, npoints):
    
    beta_min = mean - width
    beta_max = mean + width

    beta = np.linspace(beta_min,beta_max, npoints)    
    return beta

def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n




##########################################################################################
def approx_markov(rho, sigma_u, m=3, n=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian
    process with zero mean.

    Parameters
    ----------
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    x : array_like(float, ndim=1)
        The state space of the discretized process
    P : array_like(float, ndim=2)
        The Markov transition matrix where P[i, j] is the probability
        of transitioning from x[i] to x[j]

    """
    F = norm(loc=0, scale=sigma_u).cdf

    # standard deviation of y_t
    std_y = np.sqrt(sigma_u**2 / (1-rho**2))

    # top of discrete state space
    x_max = m * std_y

    # bottom of discrete state space
    x_min = - x_max

    # discretized state space
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    for i in range(n):
        P[i, 0] = F(x[0]-rho * x[i] + half_step)
        P[i, n-1] = 1 - F(x[n-1] - rho * x[i] - half_step)
        for j in range(1, n-1):
            z = x[j] - rho * x[i]
            P[i, j] = F(z + half_step) - F(z - half_step)

    return x, P


def IneqStat(out, nPoints, a_lb):
    

    weight = out['D'].flatten()
    #weight = np.reshape(out['D'], (nPoints[0] * nPoints[1] * nPoints[2]))
    wealthdata = out['a'].flatten()
    
    #wealthdata = np.reshape(out['a'], (nPoints[0] * nPoints[1] * nPoints[2]))

    p10 = weighted_quantile(wealthdata, 0.1,  sample_weight=weight)
    p50 = weighted_quantile(wealthdata, 0.5,  sample_weight=weight)
    p90 = weighted_quantile(wealthdata, 0.9,  sample_weight=weight)

    dummy_p10 = np.nonzero(wealthdata <= p10)
    dummy_p50 = np.nonzero(wealthdata <= p50)
    dummy_p90 = np.nonzero(wealthdata > p90)
    

    
    
    sBot =  np.sum( weight[dummy_p50]  * wealthdata[dummy_p50] )/  np.sum(weight * wealthdata)  
    #print(sBot*100) # 5%
    sTop = np.sum( weight[dummy_p90]  * wealthdata[dummy_p90] )/  np.sum(weight * wealthdata)  
    #print(sTop*100) # 50%
    sMiddle = 1 - sTop - sBot    
    
    s10 = np.sum( weight[dummy_p10]  * wealthdata[dummy_p10] )/  np.sum(weight * wealthdata)  
    
    
    # share of borrowing constrained 
    a_pop = out['a'] 
    sc = np.nonzero(a_pop < a_lb + 1E-7)
    sborrow_con = sum(out['D'][sc])
    
    # Share with negative assets 

    iconst = np.nonzero(out['a'] < 0)
    sneg = np.sum( out['D'][iconst].flatten() )
                      
    return sBot, sMiddle, sTop, s10, sborrow_con, sneg




def wKernelDens(data, weights):
    
    kde1= KDEUnivariate(data)
    kde1.fit(kernel = "gau", weights=weights, 
             bw="silverman",
             fft=False)
    
    sup = np.linspace(min(data), max(data), num=300)
    
    y = [kde1.evaluate(xi) for xi in sup]   
    
    return plt.plot(sup,y / np.sum(y) , '-')

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def weighted_percentileofscore(values, weights=None, values_sorted=False):
    """ Similar to scipy.percentileofscore, but supports weights.
    :param values: array-like with data.
    :param weights: array-like of the same length as `values`.
    :param values_sorted: bool, if True, then will avoid sorting of initial array.
    :return: numpy.array with percentiles of sorted array.
    """
    values = np.array(values)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    total_weight = weights.sum()
    return 100 * np.cumsum(weights) / total_weight


def weighted_percentile_of_score(a, weights, score, kind='weak'):
    npa = np.array(a)
    npw = np.array(weights)

    if kind == 'rank':  # Equivalent to 'weak' since we have weights.
        kind = 'weak'

    if kind in ['strict', 'mean']:
        indx = npa < score
        strict = 100 * sum(npw[indx]) / sum(weights)
    if kind == 'strict':
        return strict

    if kind in ['weak', 'mean']:    
        indx = npa <= score
        weak = 100 * sum(npw[indx]) / sum(weights)
    if kind == 'weak':
        return weak

    if kind == 'mean':
        return (strict + weak) / 2 


def unif_beta(nBeta, mid_beta, beta_var):
    
    beta = np.linspace(mid_beta-beta_var, mid_beta+beta_var, num = nBeta)
    
    
    return beta


def beta_dist(nBeta, mid_beta, beta_var, beta_disp, dist_type):
    
    if dist_type == "Uniform" :
        beta = unif_beta(nBeta, mid_beta, beta_var) 
        pi_beta = np.empty([nBeta])
        pi_beta = 1/nBeta
        

    if dist_type == "Normal" :
        beta, Pi_beta = approx_markov(0.003, beta_var/2, m=3, n=nBeta)
        pi_beta = utils.stationary(Pi_beta, pi_seed=None, tol=1E-11, maxit=10_000)
        beta +=  mid_beta
    
    if dist_type == "LeftSkewed" :
        if nBeta > 1:
            x = non_lin_range(mid_beta-beta_var, mid_beta+beta_var, nBeta, 1.3)   
            beta = - x + abs(min(-x)) + min(x) 
            beta = np.flip(beta)
            
            
            nb = 1 + np.arange(nBeta)
            pop = np.empty([nBeta])
            for j in range(nBeta):
                pop[j] = nb[j]**(10*beta_disp-0.5)
                
            
            pi_beta =   pop/np.sum(pop) 
        else:
            beta = np.empty([1])
            beta[0] = mid_beta
            pi_beta = 1
        
    if dist_type == "LogNormal" :
        x, pi_beta, Pi_beta = utils.markov_rouwenhorst(rho=1, sigma = beta_var_guess, N=nBeta)
        x *= beta_mid_guess       
        beta = - x + abs(min(-x)) + min(x) 
          
    
    Pi_beta = np.identity(nBeta)
    return beta, pi_beta, Pi_beta 

def non_lin_range(lb, ub, n, disp):
    
    if disp >= 1 :
        y = nonlinspace(lb, ub, n, disp)     
    elif   disp < 1:
        disp_ = abs(1-disp)+1
        y = np.flip(ub * lb / nonlinspace(lb, ub, n, disp_))

    return  y 

def init_search():
    
    p1 = 6  # i
    p2 = 6  # j
    p3 = 10 # k
    
    beta_mid_grid_init = np.linspace(0.5,0.93)
    beta_var_grid_init = np.linspace(0.002,0.3)
    beta_disp_grid = np.linspace(-3,3)
    
    beta_mid_grid = np.empty([p1])
    beta_var_grid = np.empty([p2])
    
    for i in range(p1):
        for j in range(p2):
            beta__, _ ,_ = beta_dist(nBeta, beta_mid_grid_init[i], beta_var_grid_init[j], 0.5, dist_type)
            if max(beta__) < 0.985/(1+r):
                beta_mid_grid[i] = beta_mid_grid_init[i]
                beta_var_grid[j] = beta_var_grid_init[j]
                
    beta_mid = beta_mid_grid[beta_mid_grid.astype(bool)]
    beta_var = beta_var_grid[beta_var_grid.astype(bool)]
    
    obj =  np.empty([len(beta_mid) * len(beta_var) * p3])
    
    count = 0 
    obj = 10 
    
    for i in range(len(beta_mid)):
        for j in range(len(beta_var)):   
            for k in range(p3):

                obj_old = obj 
                
                beta, pi_beta, Pi_beta = beta_dist(nBeta, beta_mid[i], beta_var[j], beta_disp_grid[k], dist_type )                                      
                Pi   =  np.kron(Pi_beta, Pi_ern)
                pi_e =  np.kron(pi_beta, pi_ern)

                out =   EGMhousehold.ss(EnVa = EnVa, EuVa = EuVa, Pi = Pi, Pi_ern = Pi_ern, a_grid = a_grid, Pi_seed = pi_e,
                                       e_grid = e_grid,  pi_e = pi_e, pi_ern = pi_ern, w = w  , ra = ra, beta = beta, eis = eis,
                                       q = q, N = N, destr = destr, b = b, T_dist = T_dist,
                                       div = div, lumpsum_T = lumpsum_T, ssAvgInc = ssAvgInc, VAT = VAT, pi_p = pi_p,
                                       nPoints = nPoints, ssflag=True)
                
                sBot, sMiddle, sTop, s10, sborrow_con = IneqStat(out, nPoints)
                obj = np.sqrt((out['A'] - B )**2   +  ((sBot- 0.05))**2 + 4 * abs(((sborrow_con - 0.1))))   
                print(obj)
                if obj < obj_old:
                    beta_mid_opt  = beta_mid[i]
                    beta_var_opt  = beta_var[j]
                    beta_disp_opt = beta_disp_grid[k]
                
                
    return  beta_mid_opt, beta_var_opt, beta_disp_opt, obj    



def Income_gini(x, *args):
    
    (U_inc, w, pi_ern, e_grid, N, div , lumpsum_T, ssAvgInc, cs_avg, transfers, avgTaxf) = args
    
    D = np.stack((pi_ern * N, (1-N) * pi_ern), axis=-1)
    D = D.flatten()
    
    T = transfers(pi_ern, lumpsum_T, e_grid, x)
    taxes_N =  avgTaxf(w   * e_grid, ssAvgInc, cs_avg) * w * e_grid   
    taxes_S =  avgTaxf(U_inc, ssAvgInc, cs_avg) * U_inc    
    taxes = np.stack((taxes_N, taxes_S), axis=-1)
    taxes = taxes.flatten()
    Inc = np.stack((w * e_grid + T, U_inc + T), axis=-1)
    Inc = Inc.flatten() 
    
    I_gini_pretax = gini( Inc   , w = D )       
    I_gini_aftertax = gini( Inc - taxes , w = D )   
  

    return abs(I_gini_aftertax-0.25)  + abs(I_gini_pretax-0.44)

def Income_gini_disp(x, args):
    
    (b, w, pi_ern, e_grid, N, div , lumpsum_T, ssAvgInc, cs_avg, transfers, avgTaxf) = args
    
    D = np.stack((pi_ern * N, (1-N) * pi_ern), axis=-1)
    D = D.flatten()
    
    T = transfers(pi_ern, lumpsum_T, e_grid, x)
    taxes_N =  avgTaxf(w   * e_grid, ssAvgInc, cs_avg) * w * e_grid   
    taxes_S =  avgTaxf(b   * e_grid, ssAvgInc, cs_avg) * b * e_grid     
    taxes = np.stack((taxes_N, taxes_S), axis=-1)
    taxes = taxes.flatten()
    Inc = np.stack((w * e_grid + T, b * e_grid + T), axis=-1)
    Inc = Inc.flatten() 
    
    I_gini_pretax = gini( Inc   , w = D )       
    I_gini_aftertax = gini( Inc - taxes , w = D )   
  
    #print('Pre-tax gini', I_gini_pretax, 0.44, 'After tax gini', I_gini_aftertax, 0.25)
    return print('Pre-tax gini', I_gini_pretax, 0.44, 'After tax gini', I_gini_aftertax, 0.25)


def Unemp_benefit(Benefit_type, b, e_grid, pi_ern, ne, ssw):

   # 'Prop'
   #  'Uniform'
   if Benefit_type == 0: # 'proportional'
        U_inc     = b  * e_grid  
        U_inc_agg = np.vdot(U_inc, pi_ern)
   elif Benefit_type == 1: # 'Uniform'
        U_inc     = np.repeat(b, ne)
        U_inc_agg = b
   elif Benefit_type == 2:   #'Prop_kink'
        U_inc     = np.repeat(b, ne)
        U_inc_agg = b
        for x in range(0,ne):
            limit = ssw * b
            # if b * e_grid < limit:
            #     U_inc[x] = b * e_grid
            # else: 
            #     U_inc[x] = limit
            U_inc = np.minimum( b * e_grid, limit   )
        U_inc_agg = np.vdot(U_inc, pi_ern)
        
   return  U_inc, U_inc_agg    
   
def b_calib(x, *args):
    b = x
    
    (Benefit_type, e_grid, pi_ern, ne, w, b_ratio, N, cs_avg, avgTaxf) = args 
    
    U_inc, U_inc_agg   = Unemp_benefit(Benefit_type, b, e_grid, pi_ern, ne, w)
    ssAvgInc = N * w + (1-N) * U_inc_agg
    
    
    Inc_after_T_N = np.vdot(w * e_grid *(1- avgTaxf(w   * e_grid, ssAvgInc, cs_avg)) , pi_ern)
    Inc_after_T_S = np.vdot(U_inc *(1- avgTaxf(U_inc, ssAvgInc, cs_avg) ) , pi_ern)
     
    
    res =     Inc_after_T_S / Inc_after_T_N - b_ratio

    return  res

def ern_func(x, a, b, c, d):
    return a + b * x + c * x**2 - d * 2.7**x 

def Earnings_risk_est():             
    from scipy.optimize import curve_fit

    df = pd.read_excel (r'tax_functions/Male_earnings_betas.xlsx')
    
    Elasticity = df[["y"]].to_numpy()   
    perc       = df[["x"]].to_numpy()  

    popt, pcov = curve_fit(ern_func, perc[:,0], Elasticity[:,0])
    
    return popt


def Calib_incidence(e_grid, pi_ern):    
    ern_par = Earnings_risk_est() 
    e_perc = weighted_percentileofscore(e_grid, weights=pi_ern, values_sorted=True)
    Y_ela = ern_func(e_perc, *ern_par)
    Y_ela = Y_ela - np.average(Y_ela)
    #Y = 1
    #f_con = 1 / np.sum( e_grid * pi_ern *Y ** Y_ela )
    #Incidence = f_con * Y**Y_ela
    
    return Y_ela

def create_deciles_index(wealthdata, weight, A_dist):
    
    wealthdata = wealthdata.flatten()
    weight = weight.flatten()
    
    p10 = weighted_quantile(wealthdata, 0.1,  sample_weight=weight)
    p20 = weighted_quantile(wealthdata, 0.2,  sample_weight=weight)
    p30 = weighted_quantile(wealthdata, 0.3,  sample_weight=weight)
    p40 = weighted_quantile(wealthdata, 0.4,  sample_weight=weight)    
    p50 = weighted_quantile(wealthdata, 0.5,  sample_weight=weight)    
    p60 = weighted_quantile(wealthdata, 0.6,  sample_weight=weight)       
    p70 = weighted_quantile(wealthdata, 0.7,  sample_weight=weight)
    p80 = weighted_quantile(wealthdata, 0.8,  sample_weight=weight)
    p90 = weighted_quantile(wealthdata, 0.9,  sample_weight=weight)

    p10_index = np.nonzero(A_dist <= p10)   
    p20_index = np.nonzero(np.logical_and(A_dist>p10, A_dist<=p20))
    p30_index = np.nonzero(np.logical_and(A_dist>p20, A_dist<=p30))
    p40_index = np.nonzero(np.logical_and(A_dist>p30, A_dist<=p40)) 
    p50_index = np.nonzero(np.logical_and(A_dist>p40, A_dist<=p50))
    p60_index = np.nonzero(np.logical_and(A_dist>p50, A_dist<=p60))    
    p70_index = np.nonzero(np.logical_and(A_dist>p60, A_dist<=p70))    
    p80_index = np.nonzero(np.logical_and(A_dist>p70, A_dist<=p80))    
    p90_index = np.nonzero(np.logical_and(A_dist>p80, A_dist<=p90))    
    p100_index = np.nonzero(A_dist>p90)   
     
    deciles_index =  [p10_index, p20_index, p30_index, p40_index, p50_index, p60_index,  p70_index, p80_index,p90_index,p100_index ]

    return deciles_index

def create_deciles_index(wealthdata, weight, A_dist):
    
    wealthdata = wealthdata.flatten()
    weight = weight.flatten()

    p10 = weighted_quantile(wealthdata, 0.1,  sample_weight=weight)
    p20 = weighted_quantile(wealthdata, 0.2,  sample_weight=weight)
    p30 = weighted_quantile(wealthdata, 0.3,  sample_weight=weight)
    p40 = weighted_quantile(wealthdata, 0.4,  sample_weight=weight)    
    p50 = weighted_quantile(wealthdata, 0.5,  sample_weight=weight)    
    p60 = weighted_quantile(wealthdata, 0.6,  sample_weight=weight)       
    p70 = weighted_quantile(wealthdata, 0.7,  sample_weight=weight)
    p80 = weighted_quantile(wealthdata, 0.8,  sample_weight=weight)
    p90 = weighted_quantile(wealthdata, 0.9,  sample_weight=weight)

    p10_index = np.nonzero(A_dist <= p10)   
    p20_index = np.nonzero(np.logical_and(A_dist>p10, A_dist<=p20))
    p30_index = np.nonzero(np.logical_and(A_dist>p20, A_dist<=p30))
    p40_index = np.nonzero(np.logical_and(A_dist>p30, A_dist<=p40)) 
    p50_index = np.nonzero(np.logical_and(A_dist>p40, A_dist<=p50))
    p60_index = np.nonzero(np.logical_and(A_dist>p50, A_dist<=p60))    
    p70_index = np.nonzero(np.logical_and(A_dist>p60, A_dist<=p70))    
    p80_index = np.nonzero(np.logical_and(A_dist>p70, A_dist<=p80))    
    p90_index = np.nonzero(np.logical_and(A_dist>p80, A_dist<=p90))    
    p100_index = np.nonzero(A_dist>p90)   
     
    deciles_index =  [p10_index, p20_index, p30_index, p40_index, p50_index, p60_index,  p70_index, p80_index,p90_index,p100_index ]

    return deciles_index

def create_deciles_index_single(wealthdata, weight, A_dist, dec):
    
    wealthdata = wealthdata.flatten()
    weight = weight.flatten()
    p = weighted_quantile(wealthdata, dec,  sample_weight=weight)
    pubber = weighted_quantile(wealthdata, dec+0.1,  sample_weight=weight)
 
    p_index = np.nonzero(np.logical_and(A_dist>p, A_dist<=pubber))

    return p_index



def find_perc_in_dist(data, weight, point):
    res = lambda x :  (point - weighted_quantile(data, x,  sample_weight=weight))**2
    #perc = optimize.fsolve(res, 0.5, factor = 0.1)
    bnds = [0, 1]
    res = optimize.minimize_scalar(res, 0.5, method='Bounded',  bounds = bnds)      
    
    
    return res.x


def N_specific_var(var, ss):  
    nPoints = ss['nPoints']
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2
    if var in ss:
        var_reshp = np.reshape(ss[var], (nBeta * ne, nN, nA)) 

    
    N = var_reshp[:,0,:]
    U = var_reshp[:,1,:]     
    return N, U

def N_specific_var_td(var, ss, td):  
    nPoints = ss['nPoints']
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2
    T = td[var].shape[0]
    
    var_reshp = np.reshape(td[var], (T, nBeta * ne, nN, nA)) 
    
    N = var_reshp[:,:,0,:]
    U = var_reshp[:,:,1,:]     
    return N, U



def N_mult(var, ss, td, dN):  
    nPoints = ss['nPoints']
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2
    
    T = td[var].shape[0]
    
    var_reshp = np.reshape(td[var], (T, nBeta * ne, nN, nA)) 
    
    dN_ = dN / ss['N']
    dU_ = (1- dN) / (1-ss['N'])
    #dN_ = dN - ss['N']
    #dU_ = (1- dN) / (1-ss['N'])    
    var_reshp[:,:,0,:] = var_reshp[:,:,0,:] * dN_[:, np.newaxis, np.newaxis]
    var_reshp[:,:,1,:] = var_reshp[:,:,1,:] * dU_[:, np.newaxis, np.newaxis]
    #tempU = var_reshp[:,:,1,:]
    #tempN = var_reshp[:,:,0,:]
    
    #iconst = np.nonzero(tempU < 0)
    #tempN[iconst] -= tempU[iconst]
    #tempU[iconst] = 0
    #var_reshp[:,:,1,:] = tempU
    #var_reshp[:,:,0,:] = tempN
    
    Dtd =  np.reshape(var_reshp, (T, nBeta * ne*nN, nA)) 
    #print(np.sum(Dtd[1,:,:].flatten()))
    return Dtd

def N_mult_het(ss, td, dN, dec):  
    
    nPoints = ss['nPoints']
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2

    T = td['D'].shape[0]
   
    aN, aU = N_specific_var('a', ss)
    decilesN = create_deciles_index_single(ss['a'], ss['D'], aN.flatten(), dec)
    decilesU = create_deciles_index_single(ss['a'], ss['D'], aU.flatten(), dec)
    
    var_reshp = np.reshape(td['D'], (T, nBeta * ne, nN, nA)) 
    
    dN_ = dN / ss['N'] - 1
    dU_ = (1-dN) / (1-ss['N']) -1 
    DN_temp = var_reshp[:,:,0,:]
    DN_temp = np.reshape(DN_temp, (T, nBeta * ne * nA)) 
    DU_temp = var_reshp[:,:,1,:]
    DU_temp = np.reshape(DU_temp, (T, nBeta * ne * nA)) 
    
    
    dN_temp = dN_[:,np.newaxis] + DN_temp - DN_temp
    dU_temp = dU_[:,np.newaxis] + DN_temp - DN_temp

    DN_temp[:,decilesN] = DN_temp[:,decilesN] + DN_temp[:,decilesN] * dN_temp[:,decilesN]
    DU_temp[:,decilesN] = DU_temp[:,decilesN] - DN_temp[:,decilesN] * dN_temp[:,decilesN] 
 
    DU_temp[:,decilesU] = DU_temp[:,decilesU] + DU_temp[:,decilesU] * dU_temp[:,decilesU]  
    DN_temp[:,decilesU] = DN_temp[:,decilesU] - DU_temp[:,decilesU] * dU_temp[:,decilesU]  
    
        
    
    var_reshp[:,:,0,:] = np.reshape(DN_temp, (T, nBeta * ne, nA))  
    var_reshp[:,:,1,:] = np.reshape(DU_temp, (T, nBeta * ne, nA))   
    
    Dtd = np.reshape(var_reshp, (T, nBeta * ne*nN, nA)) 
    #print(sum(Dtd[1,:,:].flatten()))

    return Dtd



