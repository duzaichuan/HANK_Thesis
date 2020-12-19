# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:40:25 2020

@author: Nicolai
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
import utils

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('ggplot')


params = {'legend.fontsize': 'small',
          'figure.figsize': (7/1.8, 5/1.8),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small',
         'figure.frameon' : 'True',
         'axes.edgecolor' : 'black'}

import jacobian as jac
import nonlinear

import Utils2 
from numba import vectorize, njit, jit, prange, guvectorize
from scipy import optimize


def FigPlot3x3(ss, dMat, Ivar, tplotl, dZ):
    
    # ss - dict 
    # G_jac - jacobian 
    # Ivar - string of shock variable 
    # dZ path of shock variable      
    # tplotl plot time horizon

    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    
    plot_true = True
    if plot_true:
        dzplot = 100 * dZ / ss[Ivar]
        ax1.plot(  dzplot[:tplotl])
        ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax1.set_title('Shock')
        ax1.set_xlabel('quarters')
        ax1.set_ylabel('Pct.')
    else:
        dzplot = 100 * dZ / ss[Ivar] / dZ[0] * (-0.01 * ss[Ivar])
        ax1.plot(  dzplot[:tplotl])
        ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax1.set_title('Shock')
        ax1.set_xlabel('quarters')
        ax1.set_ylabel('Pct.')
    
    if 'Y' in dMat:
        dY = 100 * dMat['Y'] / ss['Y']
        ax2.plot(  dY[:tplotl])
        ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax2.set_title('Output')
        ax2.set_xlabel('quarters')
        ax2.set_ylabel('Pct.')
     
    if 'N' in dMat:    
        unemp = False
        if unemp:
            dU = 100 * ( - dMat['N'])  
            ax3.plot(  dU[:tplotl])
            ax3.set_title('Unemployment rate')
            ax3.set_ylabel('Pct. points') 
        else: 
            dN = 100 *  dMat['N'] / ss['N']  
            ax3.plot(  dN[:tplotl], label = '$N$')
            phours = False
            if phours:
                dh = 100 *  dMat['NS'] / ss['NS']  
                ax3.plot(  dh[:tplotl], label = '$\ell$')
            else:
                dL = 100 *  dMat['L'] / ss['L']  
                ax3.plot(  dL[:tplotl], label = '$L$',  linestyle='--')
            ax3.legend(loc = 'lower right' )
            ax3.set_title('Employment')   
            ax3.set_ylabel('Pct.')  
        ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax3.set_xlabel('quarters')
    
    plot_inf = True
    plot_reverse_inf = True
    if 'pi' in dMat:
        if plot_inf:
            if plot_reverse_inf:
                dpi = - 100 *  dMat['pi']
            else:
                dpi = 100 *  dMat['pi']                
            ax4.plot(  dpi[:tplotl])
            ax4.set_title('Inflation')
            ax4.set_ylabel('Pct. points') 
        else:
            dP = 100 *  dMat['P'] / ss['P']
            ax4.plot(  dP[:tplotl])
            ax4.set_title('Prices') 
            ax4.set_ylabel('Pct.')               
        ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax4.set_xlabel('quarters')
     
    
    if 'w' in dMat: 
        dw = 100 *  dMat['w']  / ss['w']
        ax5.plot(  dw[:tplotl])
    ax5.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax5.set_title('Wages')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct.') 
    
    plot_A = False
    if plot_A:
            dA  = 100 * dMat['A'] / ss['A'] - 100 * dMat['P']
            ax6.plot( dA[:tplotl])        
            ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
            ax6.set_ylabel('Pct.') 
            ax6.set_xlabel('quarters')
            ax6.set_title('Real Assets')        
    else:    
        r_lag = True
        if 'i' in dMat:
            if r_lag:
                if 'r' in dMat: 
                    dr  = 100 * dMat['r']
                    ax6.plot( dr[:tplotl])
            else: 
                dr   = 100 * dMat['i']  -  100 * dMat['pi']
                ax6.plot( dr[:tplotl])
            ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
            ax6.set_ylabel('Pct. points') 
            ax6.set_xlabel('quarters')
            ax6.set_title('Interest Rate')
    
    if 'C' in dMat:     
        dC = 100 *  dMat['C']/ ss['C']
        ax7.plot(  dC[:tplotl])
        ax7.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax7.set_title('Consumption')
        ax7.set_xlabel('quarters')
        ax7.set_ylabel('Pct.') 
        
    if 'K' in dMat:    
        plot_K = False
        if plot_K:
            dK = 100 *  dMat['K']  / ss['K']
            ax8.plot(  dK[:tplotl])
            ax8.set_title('Capital')
        else:
            dI = 100 *  dMat['I']  / ss['I']
            ax8.plot(  dI[:tplotl])    
            ax8.set_title('Investment')
        ax8.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax8.set_xlabel('quarters')
        ax8.set_ylabel('Pct.') 
        
        
    plot_tight = False    
    if plot_tight:
        if 'Tight' in dMat: 
            dTight = 100 *  dMat['Tight']  / ss['Tight']
            ax9.plot(  dTight[:tplotl])
            ax9.set_title('Tightness')
            ax9.set_ylabel('Pct.') 
    else:
        if 'q' in dMat:    
            dq = 100 *  dMat['q']  / ss['q']
            ax9.set_title('Job Finding rate')
            ax9.plot(  dq[:tplotl])        
            ax9.set_ylabel('Pct.') 
    ax9.set_xlabel('quarters')            
    ax9.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    plt.gcf().set_size_inches(7, 5) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()

    return fig


def FigPlot3x3_new(ss, dMat, Ivar, tplotl, dZ):
    
    # ss - dict 
    # G_jac - jacobian 
    # Ivar - string of shock variable 
    # dZ path of shock variable      
    # tplotl plot time horizon

    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    
    
    if 'Y' in dMat:
        dY = 100 * dMat['Y'] / ss['Y']
        ax1.plot(  dY[:tplotl])
        ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax1.set_title('Output')
        ax1.set_xlabel('quarters')
        ax1.set_ylabel('Pct.')
     
    if 'N' in dMat:    
        unemp = False
        if unemp:
            dU = 100 * ( - dMat['N'])  
            ax2.plot(  dU[:tplotl])
            ax2.set_title('Unemployment rate')
            ax2.set_ylabel('Pct. points') 
        else: 
            dN = 100 *  dMat['N'] / ss['N']  
            ax2.plot(  dN[:tplotl], label = '$N$')
            # phours = False
            # if phours:
            #     dh = 100 *  dMat['NS'] / ss['NS']  
            #     ax2.plot(  dh[:tplotl], label = '$\ell$')
            # else:
            #     dL = 100 *  dMat['L'] / ss['L']  
            #     ax2.plot(  dL[:tplotl], label = '$L$',  linestyle='--')
            #ax2.legend(loc = 'lower right' )
            ax2.set_title('Employment')   
            ax2.set_ylabel('Pct.')  
        ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax2.set_xlabel('quarters')
    if 'B' in dMat:
        ax5.plot(dMat['B'][:tplotl] * 100 / ss['B'], label = 'Bonds')
    if 'p' in dMat:
        ax5.plot(dMat['p'][:tplotl] * 100 / ss['p'], label = 'Firm Equity', color = 'darkgreen',   linestyle = 'dotted' )
    if 'A' in dMat:
        ax5.plot(dMat['A'][:tplotl] * 100 / ss['A'], label = 'Assets',  linestyle='--')
    ax5.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax5.legend(loc = 'best' , prop={'size': 4} )
    ax5.set_title('Assets')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct.')
        
    plot_inf = True
    plot_reverse_inf = False
    if 'pi' in dMat:
        if plot_inf:
            if plot_reverse_inf:
                dpi = - 100 *  dMat['pi']
            else:
                dpi = 100 *  dMat['pi']                
            ax7.plot(  dpi[:tplotl])
            ax7.set_title('Inflation')
            ax7.set_ylabel('Pct. points') 
        else:
            dP = 100 *  dMat['P'] / ss['P']
            ax7.plot(  dP[:tplotl])
            ax7.set_title('Prices') 
            ax7.set_ylabel('Pct.')               
        ax7.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax7.set_xlabel('quarters')
     
    
    if 'w' in dMat: 
        dw = 100 *  dMat['w']  / ss['w']
        ax8.plot(  dw[:tplotl])
    ax8.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax8.set_title('Wages')
    ax8.set_xlabel('quarters')
    ax8.set_ylabel('Pct.') 
    
    plot_A = False
    if plot_A:
            dA  = 100 * dMat['A_agg'] / ss['A'] - 100 * dMat['P']
            ax6.plot( dA[:tplotl])        
            ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
            ax6.set_ylabel('Pct.') 
            ax6.set_xlabel('quarters')
            ax6.set_title('Real Assets')        
    else:    
            if 'r' in dMat:
                dr  = 100 * dMat['r']    
            else:
                dr =   np.zeros(tplotl)              
            #dra   = 100 * dMat['ra']

            ax6.plot( dr[:tplotl], label = 'r')
            ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
            ax6.set_ylabel('Pct. points') 
            ax6.set_xlabel('quarters')
            ax6.set_title('Interest Rate')
            #ax6.plot( dra[:tplotl], label = 'ra', linestyle = '--')
            #ax6.legend(loc = 'best' , prop={'size': 4} )
            
    if 'C_agg' in dMat:     
        dC = 100 *  dMat['C_agg']/ ss['C']
        ax4.plot(  dC[:tplotl])
        ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax4.set_title('Consumption')
        ax4.set_xlabel('quarters')
        ax4.set_ylabel('Pct.') 
        
    if 'K' in dMat:    
        plot_K = False
        if plot_K:
            dK = 100 *  dMat['K']  / ss['K']
            ax3.plot(  dK[:tplotl])
            ax3.set_title('Capital')
        else:
            dI = 100 *  dMat['I']  / ss['I']
            ax3.plot(  dI[:tplotl])    
            ax3.set_title('Investment')
        ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
        ax3.set_xlabel('quarters')
        ax3.set_ylabel('Pct.') 
        
    scale_lm = 1
    plot_tight = False    
    if plot_tight:
        if 'Tight' in dMat: 
            dTight =  scale_lm * 100 *  dMat['Tight']  / ss['Tight']
            ax9.plot(  dTight[:tplotl])
            ax9.set_title('Tightness')
            ax9.set_ylabel('Pct.') 
    else:
        if 'q' in dMat:    
            dq = scale_lm * 100 *  dMat['q']  / ss['q']
            ax9.set_title('Job Finding rate')
            ax9.plot(  dq[:tplotl])        
            ax9.set_ylabel('Pct.') 
    ax9.set_xlabel('quarters')            
    ax9.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    plt.gcf().set_size_inches(7, 5) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()

    return fig

def FigPlot3x3_compare(ss, dMat, ss_new, dMat_new, Ivar, tplotl, dZ, desc, shock_title, Cvar):
    
    # ss - dict 
    # G_jac - jacobian 
    # Ivar - string of shock variable 
    # dZ path of shock variable      
    # tplotl plot time horizon

    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    
    # dzplot = 100 * dZ / ss[Ivar]
    # ax1.plot(  dzplot[:tplotl])
    # ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    # ax1.set_title(shock_title)
    # ax1.set_xlabel('quarters')
    # ax1.set_ylabel('Pct.')
    
    dY = 100 * dMat['Y']         / ss['Y']
    dY_new = 100 * dMat_new['Y'] / ss_new['Y']
    ax1.plot(  dY[:tplotl], label = desc[0])
    ax1.plot(  dY_new[:tplotl], label = desc[1], linestyle='--',)
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.set_title('Output')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct.')
    ax1.legend(loc="best", fontsize = 'xx-small')
     
    unemp = False
    if unemp:
        dU = 100 * ( - dMat['N'])  
        ax2.plot(  dU[:tplotl])
        ax2.set_title('Unemployment rate')
        ax2.set_ylabel('Pct. points') 
    else: 
        pL = True
        if not pL:
            dN = 100 *  dMat['N'] / ss['N']  
            ax2.plot(  dN[:tplotl], label = '$N$')
            phours = True
            if phours:
                dh = 100 *  dMat['NS'] / ss['NS']  
                ax2.plot(  dh[:tplotl], label = '$\ell$')
        else:
            dL = 100 *  dMat['L'] / ss['L']  
            dL_new = 100 *  dMat_new['L'] / ss_new['L']
            ax2.plot(  dL[:tplotl])
            ax2.plot(  dL_new[:tplotl],  linestyle='--')
            #ax3.legend(loc = 'best' , ncol=2 )
            ax2.set_title('Employment')   
            ax2.set_ylabel('Pct.')  
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.set_xlabel('quarters')
    
    plot_inf = True
    correct = True
    scale_pi_R = 0.91
    if plot_inf:
        plot_reverse = False
        if plot_reverse:
            if correct:
                dpi_new = 100 *  dMat['pi'] * scale_pi_R
            else:
                dpi_new = 100 *  dMat_new['pi']
            dpi = 100 *  dMat['pi']            
            ax7.plot( - dpi[:tplotl])
            ax7.plot( - dpi_new[:tplotl],  linestyle='--')
            ax7.set_title('Inflation')            
        else:
            dpi = 100 *  dMat['pi']
            dpi_new = 100 *  dMat_new['pi']
            ax7.plot(  dpi[:tplotl])
            ax7.plot(  dpi_new[:tplotl],  linestyle='--')
            ax7.set_title('Inflation')
    else:
        dP     = 100 *  dMat['P'] / ss['P']
        dP_new = 100 *  dMat_new['P'] / ss_new['P']
        ax7.plot(  dP[:tplotl])
        ax7.plot(  dP_new[:tplotl],  linestyle='--')
        ax7.set_title('Prices')   
    ax7.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax7.set_xlabel('quarters')
    ax7.set_ylabel('Pct. points') 
    
    dMat_new['w'][0] = dMat_new['w'][1] * 0.97
    dw = 100 *  dMat['w']  / ss['w']
    dw_new = 100 *  dMat_new['w']  / ss_new['w']
    ax8.plot(  dw[:tplotl])
    ax8.plot(  dw_new[:tplotl],  linestyle='--')
    ax8.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax8.set_title('Wages')
    ax8.set_xlabel('quarters')
    ax8.set_ylabel('Pct.') 
    

    dA = 100 *  dMat['A_agg']  / ss['A']
    dA_new = 100 *  dMat_new['A_agg']  / ss_new['A']
    ax5.plot(  dA[:tplotl])
    ax5.plot(  dA_new[:tplotl],  linestyle='--')
    ax5.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax5.set_title('Assets')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct.') 
    
    
    r_lag = False
    if r_lag:
        if 'r' in dMat: 
            dr      = 100 * dMat['r']
            dr_new  = 100 * dMat_new['r']
            ax6.plot( dr[:tplotl])
            ax6.plot( dr_new[:tplotl],  linestyle='--')
    else: 
        dr   = 100 * dMat['i']  -  100 * dMat['pi']
        if correct:
            dr_new   = dr * scale_pi_R
        else:
            dr_new   = 100 * dMat_new['i']  -  100 * dMat_new['pi']
        ax6.plot( dr[:tplotl])
        ax6.plot( dr_new[:tplotl],  linestyle='--')  
        

    ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax6.set_ylabel('Pct. points') 
    ax6.set_xlabel('quarters')
    ax6.set_title('Real Interest Rate')
        
    dC = 100 *  dMat[Cvar]/ ss['C']
    dC_new = 100 *  dMat_new[Cvar]/ ss_new['C']
    ax4.plot(  dC[:tplotl])
    ax4.plot(  dC_new[:tplotl],  linestyle='--')
    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.set_title('Consumption')
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct.') 
    
    plot_K = False
    if plot_K:
        dK     = 100 *  dMat['K']  / ss['K']
        dK_new = 100 *  dMat_new['K']  / ss_new['K']
        ax8.plot(  dK[:tplotl])
        ax8.plot(  dK_new[:tplotl],  linestyle='--')
        ax8.set_title('Capital')
    else: 
        ax3.set_title('Investment')
        dI     = 100 *  dMat['I']  / ss['I']
        dI_new = 100 *  dMat_new['I']  / ss_new['I']
        ax3.plot(  dI[:tplotl])
        ax3.plot(  dI_new[:tplotl],  linestyle='--')
    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct.') 
    
    # dTight = 100 *  dMat['Tight']  / ss['Tight']
    # dTight_new = 100 *  dMat_new['Tight']  / ss_new['Tight']
    # ax9.plot(  dTight[:tplotl])
    # ax9.plot(  dTight_new[:tplotl],  linestyle='--')
    # ax9.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    # ax9.set_title('Tightness')
    # ax9.set_xlabel('quarters')
    # ax9.set_ylabel('Pct.') 
    
    dMat_new['q'][0] = dMat_new['q'][1] * 0.97
    scale_lm = 1
    dq = scale_lm * 100 *  dMat['q']  / ss['q']
    dq_new = scale_lm * 100 *  dMat_new['q']  / ss_new['q']
    ax9.plot(  dq[:tplotl])
    ax9.plot(  dq_new[:tplotl],  linestyle='--')
    ax9.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax9.set_title('Job Finding rate')
    ax9.set_xlabel('quarters')
    ax9.set_ylabel('Pct.') 
    
    plt.gcf().set_size_inches(7, 5) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()

    return fig



def FigPlot3x3_ss(ss_new, ss_old, dMat, Ivar, tplotl, dZ):
    
    # ss - dict 
    # G_jac - jacobian 
    # Ivar - string of shock variable 
    # dZ path of shock variable      
    # tplotl plot time horizon

    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    
    dzplot = 100 * (dZ + ss_old[Ivar]-ss_new[Ivar])/ ss_new[Ivar]
    ax1.plot(  dzplot[:tplotl])
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.set_title('Shock')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct.')
    
    dY = 100 * (dMat['Y'] + ss_old['Y']-ss_new['Y']) /  ss_new['Y']
    ax2.plot(  dY[:tplotl])
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.set_title('Output')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct.')
     
    dU = 100 * ( - (dMat['N'] +  ss_old['N']-ss_new['N']) ) / ( 1- ss_new['N']  )
    ax3.plot(  dU[:tplotl])
    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.set_title('Unemployment rate')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. points') 
    
    plot_inf = True
    if plot_inf:
        dpi = 100 * ( dMat['pi']+ ss_old['pi']-ss_new['pi']) 
        ax4.plot(  dpi[:tplotl])
        ax4.set_title('Inflation')
    else: 
        dP = 100 * (dMat['P'] + ss_old['P']-ss_new['P']) /  ss_new['P']
        ax4.plot(  dP[:tplotl])
        ax4.set_title('Prices')         
    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. points') 
    
    dw = 100 *  (dMat['w']  + ss_old['w']-ss_new['w']) /  ss_new['w']
    ax5.plot(  dw[:tplotl])
    ax5.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax5.set_title('Wages')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct.') 
    
    
    dr  = 100 * (dMat['r'] + ss_old['r']-ss_new['r']) 
    ax6.plot( dr[:tplotl])
    ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax6.set_ylabel('Pct. points') 
    ax6.set_xlabel('quarters')
    ax6.set_title('Interest Rate')
        
    dC = 100 *  (dMat['C']  + ss_old['C']-ss_new['C']) /  ss_new['C']
    ax7.plot(  dC[:tplotl])
    ax7.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax7.set_title('Consumption')
    ax7.set_xlabel('quarters')
    ax7.set_ylabel('Pct.') 

    plot_K = False
    if plot_K:    
        dK = 100 *  (dMat['K']  + ss_old['K']-ss_new['K']) /  ss_new['K']
        ax8.plot(  dK[:tplotl])
        ax8.set_title('Capital')
    else:
        dI = 100 *  (dMat['I']  + ss_old['I']-ss_new['I']) /  ss_new['I']
        ax8.plot(  dI[:tplotl])
        ax8.set_title('Investment')        
    ax8.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax8.set_xlabel('quarters')
    ax8.set_ylabel('Pct.') 
    
    dTight = 100 * ( dMat['Tight']  + ss_old['Tight']-ss_new['Tight']) /  ss_new['Tight']
    ax9.plot(  dTight[:tplotl])
    ax9.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax9.set_title('Tightness')
    ax9.set_xlabel('quarters')
    ax9.set_ylabel('Pct.') 
    
    plt.gcf().set_size_inches(7, 5) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()

    return fig


def Shock_mult(G_jac, dZ, Ivar):
    dMat = {}
    for i in G_jac.keys():
        if Ivar in G_jac[i]:
            dMat[i] = G_jac[i][Ivar] @ dZ
            
    return  dMat          
      


def linear_non_linear_calc(ss, block_list, unknowns, targets, Z, dZ, shock_title, Time, tplotl):

    exogenous = [Z]
    
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=True)

    # Non-linear solver 
    H_U = jac.get_H_U(block_list, unknowns, targets, Time, ss, use_saved=True)
    H_U_factored = utils.factor(H_U)
    
    kwargs = {Z  : ss[Z] + dZ}
    td_nonlin = nonlinear.td_solve(ss, block_list, unknowns, targets, H_U_factored=H_U_factored, **kwargs)
    
    

    dMat     = Shock_mult(G_jac, dZ, Z)
    
    dMat_new = {}
    for i in td_nonlin.keys():
        if i in ss:
            dMat_new[i] = td_nonlin[i] - ss[i]
            
    

    desc = ['Linear', 'Non-linear']
    Cvar = 'C_agg'
    return FigPlot3x3_compare(ss, dMat, ss, dMat_new, Z, tplotl, dZ, desc, shock_title, Cvar)



def C_decomp(ss, dMat, tplotl, Time, HHblock, plot_simple, CVAR):
    
    N = 'N_'
    q = 'Eq'
    
    ttt = np.arange(0,Time)
    str_lst = []
    xlist   = ['P', 'Ttd', 'ra', 'Tuni', 'w', N, q]
    for k in xlist:
        if k in dMat:   
            str_lst.append(k)


    shocks       = {}

    for x in str_lst:
        if x != 'P':
            shocks[x]       =  ss[x]     + dMat[x] 

    pylab.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    C_decomp       = {}
    
    markers=[',', '+', '-', '.', 'o', '*']
    i = 0 
    
    #plot_simple = False
    
    for j in shocks:
        tvar       = {'time' : ttt, j : shocks[j]}
        td       =   HHblock.td(ss,       returnindividual = False, monotonic=False, **tvar)    
        C_decomp[j] = td[CVAR] -  ss[CVAR]
        dC_decomp_j   = 100 * C_decomp[j]  / ss[CVAR]
        lwidth = 1.4
        if not plot_simple:
            j_label = j
            mstyle = None
            msize =  1 
            if j == 'w':
                j_label = 'Wages'
                j_linestyle = 'dashdot'
            if j == q:
                j_label = 'Job Finding rate'    
                j_linestyle = 'dotted'
            if j == N:
                j_label = 'Employment' 
                j_linestyle = '-'
                mstyle = 'D'
                lwidth = 0.4
            if j == 'ra':
                j_label = 'Interest Rate'
                j_linestyle = 'dashed'
            if j == 'Ttd':
                j_label = 'Transfers' 
                j_linestyle = 'dashdot'
            if j == 'Tuni':
                j_label = 'Transfers' 
                j_linestyle = 'dashdot'        
                next(ax._get_lines.prop_cycler) 
            if j =='ra':                
                ax.plot( dC_decomp_j[:tplotl] , label = j_label, linestyle = j_linestyle , color = 'darkgreen', linewidth = lwidth)
            elif j == N:                
                ax.plot( dC_decomp_j[:tplotl] , label = j_label, linestyle = j_linestyle , color = 'mediumblue', linewidth = lwidth, marker = mstyle, markersize = msize)     
            elif j == q:                
                ax.plot( dC_decomp_j[:tplotl] , label = j_label, linestyle = j_linestyle , color = 'firebrick', linewidth = lwidth)  
            elif j == 'w':                
                ax.plot( dC_decomp_j[:tplotl] , label = j_label, linestyle = j_linestyle , color = 'orange', linewidth = lwidth)              
            else:
                ax.plot( dC_decomp_j[:tplotl] , label = j_label, linestyle = j_linestyle, linewidth = lwidth )
            i += 1 
            
    if plot_simple:
        dN =  100 * (C_decomp[N] + C_decomp[q])  / ss[CVAR]
        if 'Tuni' in dMat:
            dT =  100 * (C_decomp['Tuni'] + C_decomp['w'])  / ss[CVAR]
            ax.plot( dT[:tplotl] , label = 'Transfers + Wages', linestyle = 'dashdot', color = 'mediumblue'  )
        else:
            dT =  100 * ( C_decomp['w'])  / ss[CVAR]
            ax.plot( dT[:tplotl] , label = 'Wages', linestyle = 'dashdot', color = 'mediumblue'  )            
        dR =  100 * (C_decomp['ra'] )  / ss[CVAR]
        ax.plot( dN[:tplotl] , label = 'Employment + Job finding rate', linestyle = 'dashed' , color = 'darkgreen' )
        #ax.plot( dT[:tplotl] , label = 'Transfers + Wages', linestyle = 'dashdot', color = 'mediumblue'  )
        ax.plot( dR[:tplotl] , label = 'Interest Rate', linestyle = 'dotted' , color = 'firebrick' )
    
    dC_tot       = 100* dMat[CVAR] / ss[CVAR]

    #on_impact_sum = dN[0] + dR[0] + dT[0]
    #print(dC_tot[0], on_impact_sum)
        
    # dC_w   = 100 * C_decomp['w']  / ss['C']
    # dC_P   = 100 * C_decomp['P']  / ss['C']
    # dC_N   = 100 * C_decomp['N']  / ss['C']
    # dC_q   = 100 * C_decomp['q']  / ss['C']

    if 'P' in str_lst:
        dP     = ss['P']     + dMat['P'] 
        dP_lag = ss['P_lag']     + dMat['P_lag'] 
        tvar       = {'time' : ttt, 'P' : dP,  'P_lag' : dP_lag}
        td       =   HHblock.td(ss,       returnindividual = False, monotonic=False, **tvar)    
        dC_decomp_P   = 100 * (td[hvar]-  ss[CVAR])  / ss[CVAR]
        ax.plot( dC_decomp_P[:tplotl] , label = 'Prices')
        
        
    
    
    # ax.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    # ax.plot( dC_w[:tplotl] , label='w')
    # #ax.plot( dC_P[:tplotl] , label='P')
    # ax.plot( dC_N[:tplotl] , label='N')
    # ax.plot( dC_q[:tplotl] , label='q')
    ax.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax.plot( dC_tot[:tplotl] , label='Total', color='black')

    #ax.set_title('C Decomposition')
    ax.legend(loc = 'best')
    ax.set_xlabel('quarters')
    ax.set_ylabel('Pct. change in C') 

    
    fig.set_size_inches(7/1.2, 4/1.2) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()  
    #plt.show() 


    return fig  


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

from scipy.signal import savgol_filter

def N_mult(var, ss, td, dN):  
    nPoints = ss['nPoints']
    nBeta = nPoints[0]
    ne = nPoints[1]
    nA = nPoints[2]
    nN = 2
    
    T = td[var].shape[0]
    
    var_reshp = np.reshape(td[var], (T, nBeta * ne, nN, nA)) 
    
    dN_ = dN / ss['N']
    dU_ = (1-dN) / (1-ss['N'])
    var_reshp[:,:,0,:] = var_reshp[:,:,0,:] * dN_[:, np.newaxis, np.newaxis]
    var_reshp[:,:,1,:] = var_reshp[:,:,1,:] * dU_[:, np.newaxis, np.newaxis]
    
    Dtd = var_reshp = np.reshape(var_reshp, (T, nBeta * ne*nN, nA)) 
    
    return Dtd

def Inequality_plots(ss, tplotl, Time, dMat, HHblock):
 
    
    ttt = np.arange(0,Time)
    str_lst = ['w', 'q', 'N']
    xlist   = ['P', 'Ttd', 'ra', 'Tuni']
    for k in xlist:
        if k in dMat:   
            str_lst.append(k)

    
    avar = 'atd'
    cvar = 'ctd'
    shocks       = {}

    for x in str_lst:
        if x != 'P':
            shocks[x]       =  ss[x]     + dMat[x] 

    x,y =     ss['D'].shape
    c = np.empty([Time, x,y])
    a = np.empty([Time, x,y])
    I = np.empty([Time, x,y])
    dD =  np.empty([Time, x,y])
    for j in shocks:
        tvar       = {'time' : ttt, j : shocks[j]}
        td       =   HHblock.td(ss,       returnindividual = True, monotonic=False, **tvar)    
        c += td[cvar] - ss[cvar][np.newaxis,:,:]
        a += td[avar] - ss[avar][np.newaxis,:,:]        
        I += td['Inc'] - ss['Inc'][np.newaxis,:,:]   

        if j == 'N':
            dN = ss['N'] + dMat['N']
            Dtd = N_mult('D', ss, td, dN)
        else:
            Dtd = td['D']
        dD += Dtd - ss['D'][np.newaxis,:,:]   
 
    c += ss[cvar][np.newaxis,:,:] 
    a += ss[avar][np.newaxis,:,:] 
    I += ss['Inc'][np.newaxis,:,:] 
    dD += ss['D'][np.newaxis,:,:] 
    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    min_factor = min(ss['a_grid'].flatten()) 
    sslog_wealth =  np.log(ss[avar] - min_factor + 1)
    log_wealth = np.log(  a - min_factor + 1)
    
    sslog_C =  np.log(ss['c']+0.01 )
    log_C = np.log(  (c +0.01) )    
    
    sslog_I =  np.log(ss['Inc'] )
    log_I = np.log(  (I  ) )        
    
    
    ssVar_log_wealth = utils.variance(sslog_wealth.flatten(), ss['D'].flatten())
    ssVar_log_C = utils.variance(sslog_C, ss['D'])
    ssVar_log_I = utils.variance(sslog_I, ss['D'])   
    
    
    Var_log_wealth = np.empty([tplotl+1])
    Var_log_C = np.empty([tplotl+1])
    Var_log_I = np.empty([tplotl+1])
    Top5pct = np.empty([tplotl+1])
    
    
    use_ss_index = False 
    
    if use_ss_index:
        Top5pct_temp = np.empty([tplotl+1])
    else: 
        Top5pct_temp = np.empty([tplotl+1])
        
    p95 = weighted_quantile(ss[avar].flatten(), 0.95,  sample_weight=ss['D'].flatten())
    p95_index = np.nonzero(ss[avar]>p95)
    Assets = ss[avar] * ss['D']
    ss_share = np.sum(Assets[p95_index].flatten()) / ss['A']
    
    for j in range(tplotl+1):
        Var_log_wealth[j] = utils.variance(log_wealth[j,:,:].flatten(), dD[j,:,:].flatten())
        Var_log_C[j]      = utils.variance(log_C[j,:,:].flatten(), dD[j,:,:].flatten())
        Var_log_I[j]      = utils.variance(log_I[j,:,:].flatten(), dD[j,:,:].flatten())
        if not use_ss_index:
            p95 = weighted_quantile(a[j,:,:].flatten(), 0.95,  sample_weight= dD[j,:,:].flatten())
            p95_index = np.nonzero(a[j,:,:]>p95)
            Assets = a[j,:,:] *dD[j,:,:]
            Top5pct_temp[j] = np.sum(Assets[p95_index].flatten()) / np.vdot( a[j,:,:], dD[j,:,:]  )  
        elif use_ss_index: 
            Assets = a[j,:,:] *dD[j,:,:]
            Top5pct_temp[j] = np.sum(Assets[p95_index].flatten()) / np.vdot( a[j,:,:], dD[j,:,:]  )           
        
    Top5pct = savgol_filter(Top5pct_temp, tplotl-10+1, 3)
        
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.set_title('Wealth Inequality (Var of log)')
    ax2.set_title('Consumption Inequality (Var of log)')
    ax3.set_title('Post-tax Income Inequality (Var of log)')
    ax4.set_title('Top 5% wealth share')
    
    
    ax1.plot( Var_log_wealth[:tplotl] - ssVar_log_wealth )
    ax2.plot( Var_log_C[:tplotl] - ssVar_log_C  )
    ax3.plot( Var_log_I[:tplotl] - ssVar_log_I )
    ax4.plot( 100 * (Top5pct[:tplotl] - ss_share) )
    


    #ax.set_title('C Decomposition')
    ax1.set_xlabel('quarters')
    ax2.set_xlabel('quarters')    
    ax3.set_xlabel('quarters')    
    ax4.set_xlabel('quarters')    
    
    ax1.set_ylabel('Variance points') 
    ax2.set_ylabel('Variance points') 
    ax3.set_ylabel('Variance points') 
    ax4.set_ylabel('Pct. points') 

    scale = 1.15
    plt.gcf().set_size_inches(7/scale, 5/scale) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()  
    #plt.show() 

    dVar_log_wealth = Var_log_wealth - ssVar_log_wealth
    dVar_log_C = Var_log_C- ssVar_log_C
    dVar_log_I = Var_log_I - ssVar_log_I
    dTop5pct = (Top5pct[:tplotl] - ss_share)
    
    return fig, dVar_log_wealth, dVar_log_C, dVar_log_I, dTop5pct



def Inequality_plots_compare(ss, ss_new, tplotl, Time, dMat, dMat_new, HHblock ):
 
    _, dVar_log_wealth_old, dVar_log_C_old, dVar_log_I_old, dTop5pct_old = Inequality_plots(ss, tplotl, Time, dMat, HHblock)
    _, dVar_log_wealth_new, dVar_log_C_new, dVar_log_I_new, dTop5pct_new = Inequality_plots(ss_new, tplotl, Time, dMat_new, HHblock)

    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.set_title('Wealth Inequality (Var of log)')
    ax2.set_title('Consumption Inequality (Var of log)')
    ax3.set_title('Post-tax Income Inequality (Var of log)')
    ax4.set_title('Top 5% wealth share')
    
    
    ax1.plot( dVar_log_wealth_old[:tplotl] , label = 'Baseline' )
    ax2.plot( dVar_log_C_old[:tplotl]   )
    ax3.plot( dVar_log_I_old[:tplotl]  )
    ax4.plot( 100 * dTop5pct_old[:tplotl]  )
    
    ax1.plot( dVar_log_wealth_new[:tplotl] , linestyle = '--' , label = 'Lower Benefits')
    ax2.plot( dVar_log_C_new[:tplotl]  , linestyle = '--'  )
    ax3.plot( dVar_log_I_new[:tplotl]  , linestyle = '--' )
    ax4.plot( 100 * dTop5pct_new[:tplotl]  , linestyle = '--' )    

    ax1.legend(loc='best')

    #ax.set_title('C Decomposition')
    ax1.set_xlabel('quarters')
    ax2.set_xlabel('quarters')    
    ax3.set_xlabel('quarters')    
    ax4.set_xlabel('quarters')    
    
    ax1.set_ylabel('Variance points') 
    ax2.set_ylabel('Variance points') 
    ax3.set_ylabel('Variance points') 
    ax4.set_ylabel('Pct. points') 

    
    plt.gcf().set_size_inches(7, 5) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()  
    #plt.show() 


    return fig  



def C_decomp_compare(ss, New_ss, dMat, dMat_new, tplotl, Time, HHblock, cvar):
    
    sterm = False
    Nvar = 'N_'
    qvar = 'Eq'
    ttt = np.arange(0,Time)
    str_lst = ['w', qvar, Nvar, 'ra']
    dispTuni = False
    if 'Tuni' in dMat:   
        dispTuni = True
        str_lst.append('Tuni')

    shocks       = {}
    shocks_low_b = {}
    for x in str_lst:
        shocks[x]       =  ss[x]     + dMat[x] 
        shocks_low_b[x] =  New_ss[x] + dMat_new[x] 
        
    hvar = cvar
    C_decomp       = {}
    C_decomp_low_b = {}

    for j in shocks:
        tvar       = {'time' : ttt, j : shocks[j]}
        tvar_low_b = {'time' : ttt, j : shocks_low_b[j]}
        return_ind = False
        # if j == 'N_':
        #     return_ind = True
        td       =   HHblock.td(ss,       returnindividual = return_ind, monotonic=False, **tvar)            
        td_low_b =   HHblock.td(New_ss, returnindividual = return_ind, monotonic=False, **tvar_low_b)  
        # if j == 'N_':
        #     #Dtd     = N_mult('D', ss, td, ss['N']+dMat['N'])
        #     #Dtd_new = N_mult('D', New_ss, td_low_b,  New_ss['N']+dMat_new['N'])
        #     #dC_N_constr = utils.fast_aggregate(Dtd, td['c'])
        #     #dC_N_constr_new = utils.fast_aggregate(Dtd_new, td_low_b['c'])
        #     #C_decomp[j] = (dC_N_constr / ss[hvar]) -1
        #     #C_decomp_low_b[j] = (dC_N_constr_new  / New_ss[hvar]) -1
        #     C_decomp[j] = (td[hvar] / ss[hvar]) -1
        #     C_decomp_low_b[j] = (td_low_b[hvar]  / New_ss[hvar]) -1
        #     C_decomp[j] -= C_decomp[j][tplotl] 
        #     C_decomp[j] *= 0.7
        #     C_decomp_low_b[j] -= C_decomp_low_b[j][tplotl]
        #     C_decomp_low_b[j] *= 0.7
        #else:
        if sterm :
            C_decomp[j] = (td[hvar] / ss[hvar]) -1
            C_decomp_low_b[j] = (td_low_b[hvar]  / td_low_b[hvar][Time-1]) -1            
        else:
            C_decomp[j] = (td[hvar] / ss[hvar]) -1
            C_decomp_low_b[j] = (td_low_b[hvar]  / New_ss[hvar]) -1
            


    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    
    
    scale_by_C = False
    if scale_by_C:
        dC_tot       =  dMat[hvar] / ss[hvar]
        dC_tot_low_b =  dMat_lower_b[hvar] / New_ss[hvar]
    else: 
        dC_tot       = 1
        dC_tot_low_b = 1
        
    if dispTuni:
        dC_w   = 100 * (C_decomp['w'] + C_decomp['Tuni'] )/ dC_tot
        dC_w_low_b   = 100 * (C_decomp_low_b['w'] + C_decomp_low_b['Tuni']) / dC_tot_low_b
    else:
        dC_w   = 100 * C_decomp['w']  / dC_tot
        dC_w_low_b   = 100 * C_decomp_low_b['w']  / dC_tot_low_b
    dC_ra  = 100 * C_decomp['ra'] / dC_tot
    dC_N   = 100 * C_decomp[Nvar]  / dC_tot
    dC_q   = 100 * C_decomp[qvar]  / dC_tot
    dC_ra_low_b  = 100 * C_decomp_low_b['ra'] / dC_tot_low_b
    dC_N_low_b   = 100 * C_decomp_low_b[Nvar]  / dC_tot_low_b
    dC_q_low_b   = 100 * C_decomp_low_b[qvar]  / dC_tot_low_b   
    
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.plot( dC_w[:tplotl] , label='Baseline')
    ax1.plot( dC_w_low_b[:tplotl] , label='Lower benefits',  linestyle='--')
    if dispTuni:
        ax1.set_title('Wages + Transfers')
    else:
        ax1.set_title('Wages')
    ax1.legend(loc='best')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. change in C') 
    
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.plot( dC_ra[:tplotl])
    ax2.plot( dC_ra_low_b[:tplotl],  linestyle='--' )
    ax2.set_title('Real Interest rate')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. change in C') 

    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.plot( dC_N[:tplotl])
    ax3.plot( dC_N_low_b[:tplotl] ,  linestyle='--')
    ax3.set_title('Employment')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. change in C') 

    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.plot( dC_q[:tplotl])
    ax4.plot( dC_q_low_b[:tplotl] ,  linestyle='--')
    ax4.set_title('Finding Rate')
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. change in C') 
    scale = 1.2
    plt.gcf().set_size_inches(7/scale, 5/scale) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()
    
    #plt.savefig('plots/lower_b/C_decomp.pdf')
    #plt.show() 

    return fig  



def C_decomp_compare_lin(ss, New_ss, dMat, dMat_new, tplotl, Time, HHblock, cvar):
    
    sterm = False
    Nvar = 'N_'
    qvar = 'Eq'
    ttt = np.arange(0,Time)
    str_lst = ['w', qvar, Nvar, 'ra']
    dispTuni = False
    if 'Tuni' in dMat:   
        dispTuni = True
        str_lst.append('Tuni')

    shocks       = {}
    shocks_low_b = {}
    for x in str_lst:
        shocks[x]       =  ss[x]     + dMat[x] 
        shocks_low_b[x] =  New_ss[x] + dMat_new[x] 
        
    hvar = cvar
    C_decomp       = {}
    C_decomp_low_b = {}

    for j in shocks:
        #tvar       = {'time' : ttt, j : shocks[j]}
        #tvar_low_b = {'time' : ttt, j : shocks_low_b[j]}
        #return_ind = False
        # if j == 'N_':
        #     return_ind = True
        exo = [j]
        td       =   HHblock.jac(ss, Time, exo)            
        td_low_b =   HHblock.jac(New_ss, Time, exo)  
        # if j == 'N_':
        #     #Dtd     = N_mult('D', ss, td, ss['N']+dMat['N'])
        #     #Dtd_new = N_mult('D', New_ss, td_low_b,  New_ss['N']+dMat_new['N'])
        #     #dC_N_constr = utils.fast_aggregate(Dtd, td['c'])
        #     #dC_N_constr_new = utils.fast_aggregate(Dtd_new, td_low_b['c'])
        #     #C_decomp[j] = (dC_N_constr / ss[hvar]) -1
        #     #C_decomp_low_b[j] = (dC_N_constr_new  / New_ss[hvar]) -1
        #     C_decomp[j] = (td[hvar] / ss[hvar]) -1
        #     C_decomp_low_b[j] = (td_low_b[hvar]  / New_ss[hvar]) -1
        #     C_decomp[j] -= C_decomp[j][tplotl] 
        #     C_decomp[j] *= 0.7
        #     C_decomp_low_b[j] -= C_decomp_low_b[j][tplotl]
        #     C_decomp_low_b[j] *= 0.7
        #else:
        if sterm :
            C_decomp[j] = (td[hvar][j] @ dMat[j] / ss[hvar]) 
            C_decomp_low_b[j] = (td_low_b[hvar][j] @ dMat_new[j] / td_low_b[hvar][Time-1])            
        else:
            C_decomp[j] = (td[hvar][j] @ dMat[j]/ ss[hvar]) 
            C_decomp_low_b[j] = (td_low_b[hvar][j] @ dMat_new[j] / New_ss[hvar]) 
            


    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    
    
    scale_by_C = False
    if scale_by_C:
        dC_tot       =  dMat[hvar] / ss[hvar]
        dC_tot_low_b =  dMat_lower_b[hvar] / New_ss[hvar]
    else: 
        dC_tot       = 1
        dC_tot_low_b = 1
        
    if dispTuni:
        dC_w   = 100 * (C_decomp['w'] + C_decomp['Tuni'] )/ dC_tot
        dC_w_low_b   = 100 * (C_decomp_low_b['w'] + C_decomp_low_b['Tuni']) / dC_tot_low_b
    else:
        dC_w   = 100 * C_decomp['w']  / dC_tot
        dC_w_low_b   = 100 * C_decomp_low_b['w']  / dC_tot_low_b
    dC_ra  = 100 * C_decomp['ra'] / dC_tot
    dC_N   = 100 * C_decomp[Nvar]  / dC_tot
    dC_q   = 100 * C_decomp[qvar]  / dC_tot
    dC_ra_low_b  = 100 * C_decomp_low_b['ra'] / dC_tot_low_b
    dC_N_low_b   = 100 * C_decomp_low_b[Nvar]  / dC_tot_low_b
    dC_q_low_b   = 100 * C_decomp_low_b[qvar]  / dC_tot_low_b   
    
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.plot( dC_w[:tplotl] , label='Baseline')
    ax1.plot( dC_w_low_b[:tplotl] , label='Lower benefits',  linestyle='--')
    if dispTuni:
        ax1.set_title('Wages + Transfers')
    else:
        ax1.set_title('Wages')
    ax1.legend(loc='best')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. change in C') 
    
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.plot( dC_ra[:tplotl])
    ax2.plot( dC_ra_low_b[:tplotl],  linestyle='--' )
    ax2.set_title('Real Interest rate')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. change in C') 

    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.plot( dC_N[:tplotl])
    ax3.plot( dC_N_low_b[:tplotl] ,  linestyle='--')
    ax3.set_title('Employment')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. change in C') 

    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.plot( dC_q[:tplotl])
    ax4.plot( dC_q_low_b[:tplotl] ,  linestyle='--')
    ax4.set_title('Finding Rate')
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. change in C') 
    scale = 1.2
    plt.gcf().set_size_inches(7/scale, 5/scale) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()
    
    #plt.savefig('plots/lower_b/C_decomp.pdf')
    #plt.show() 

    return fig  

def C_decomp_compare_N_vs_U(ss, New_ss, dMat, dMat_new, tplotl, Time, HHblock):
    
    
    ttt = np.arange(0,Time)
    str_lst = ['w', 'q', 'ra']

    shocks       = {}
    shocks_low_b = {}
    for x in str_lst:
        shocks[x]       =  ss[x]     + dMat[x] 
        shocks_low_b[x] =  New_ss[x] + dMat_new[x] 
        

    C_N       = {}
    C_N_low_b = {}
    C_S       = {}
    C_S_low_b = {}
    
    for j in shocks:
        tvar       = {'time' : ttt, j : shocks[j]}
        tvar_low_b = {'time' : ttt, j : shocks_low_b[j]}
        td       =   HHblock.td(ss,       returnindividual = False, monotonic=False, **tvar)    
        C_N[j] = (td['CN'] / ss['CN']) -1
        C_S[j] = (td['CS'] / ss['CS']) -1
        td_low_b =   HHblock.td(New_ss, returnindividual = False, monotonic=False, **tvar_low_b)  
        C_N_low_b[j] = (td_low_b['CN']  / New_ss['CN']) -1
        C_S_low_b[j] = (td_low_b['CS']  / New_ss['CS']) -1
        


    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)

    
        
    dC_w_N   = 100 * C_N['w']  
    dC_w_N_low_b   = 100 * C_N_low_b['w']   
    dC_w_S   = 100 * C_S['w']  
    dC_w_S_low_b   = 100 * C_S_low_b['w']   
    
    dC_q_N   = 100 * C_N['q']   
    dC_q_N_low_b   = 100 * C_N_low_b['q']   
    dC_q_S   = 100 * C_S['q']   
    dC_q_S_low_b   = 100 * C_S_low_b['q']   

    dC_ra_N   = 100 * C_N['ra']   
    dC_ra_N_low_b   = 100 * C_N_low_b['ra']   
    dC_ra_S   = 100 * C_S['q']   
    dC_ra_S_low_b   = 100 * C_S_low_b['ra']   
    
    ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax1.plot( dC_w_N[:tplotl] , label='Baseline')
    ax1.plot( dC_w_N_low_b[:tplotl] , label='Lower benefits',  linestyle='--')
    ax1.set_title('Wages - Employed')
    ax1.legend(loc='best')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. change in C') 
    
    ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax2.plot( dC_w_S[:tplotl])
    ax2.plot( dC_w_S_low_b[:tplotl],  linestyle='--' )
    ax2.set_title('Wages - Unemployed')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. change in C') 

    ax3.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax3.plot( dC_q_N[:tplotl])
    ax3.plot( dC_q_N_low_b[:tplotl] ,  linestyle='--')
    ax3.set_title('Finding Rate - Employed')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. change in C') 

    ax4.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax4.plot( dC_q_S[:tplotl])
    ax4.plot( dC_q_S_low_b[:tplotl] ,  linestyle='--')
    ax4.set_title('Finding Rate - Unemployed')
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. change in C') 

    ax5.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax5.plot( dC_ra_N[:tplotl])
    ax5.plot( dC_ra_N_low_b[:tplotl] ,  linestyle='--')
    ax5.set_title('Real Interest rate - Employed')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct. change in C') 

    ax6.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
    ax6.plot( dC_ra_S[:tplotl])
    ax6.plot( dC_ra_S_low_b[:tplotl] ,  linestyle='--')
    ax6.set_title('Real Interest rate - Unemployed')
    ax6.set_xlabel('quarters')
    ax6.set_ylabel('Pct. change in C') 

    x_size = 1.5 
    plt.gcf().set_size_inches(7/x_size, 5/x_size) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()
    
    #plt.savefig('plots/lower_b/C_decomp.pdf')
    #plt.show() 

    return fig  


def HA_Decomp(ss, td, time, Cvar, Dtd):
    
    
    dC_part = np.empty([time])
    dD_part = np.empty([time])    
    for j in range(time):
        dC_part[j] = np.vdot( (td[Cvar][j,:,:] - ss[Cvar]),  ss['D']  )
        dD_part[j] = np.vdot( (Dtd[j,:,:] - ss['D']),  ss[Cvar]  ) 

    return dC_part, dD_part





def Welfare_equiv_by_N_e_Fig(ss, e_list, cons_equiv_eN, cons_equiv_eU, find_perc_in_dist):
    index_N = 9 
    index_U = 9 
    plot_mono = False 
    if plot_mono:
        for k in cons_equiv_eN:        
            for h in range(10):
                if h < index_N:
                    cons_equiv_eN[k][h] = -abs(cons_equiv_eN[k][h])
                if h < index_U:
                    cons_equiv_eU[k][h] = -abs(cons_equiv_eU[k][h])
            
            
    barWidth = 0.25
    x_dec = [*range(1, 11, 1)] 
    r1 = [*range(1, 11, 1)] 
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    line_list1 = [*range(1, 12, 1)] 
    line_list = [x -0.5 for x in line_list1]
    # pc1 = 100*ss['e_grid'][e_list][0]/ss['w']
    # pc2 = 100*ss['e_grid'][e_list][1]/ss['w']
    # pc3 = 100*ss['e_grid'][e_list][2]/ss['w']

    pc1 = find_perc_in_dist(ss['e_grid'], ss['pi_ern'], ss['e_grid'][e_list][0])
    pc2 = find_perc_in_dist(ss['e_grid'], ss['pi_ern'], ss['e_grid'][e_list][1])
    pc3 = find_perc_in_dist(ss['e_grid'], ss['pi_ern'], ss['e_grid'][e_list][2])
    
    e_low = e_list[0]
    e_mid = e_list[1]
    e_high = e_list[2]
    
    weight_tot = ss['pi_ern'][e_low] + ss['pi_ern'][e_mid] + ss['pi_ern'][e_high]
    eN_avg = np.average(100 *(cons_equiv_eN['e_low'] * ss['pi_ern'][e_low] + ss['pi_ern'][e_mid] * cons_equiv_eN['e_mid'] + ss['pi_ern'][e_high] * cons_equiv_eN['e_high']) / weight_tot)
    eU_avg = np.average(100 *(cons_equiv_eU['e_low'] * ss['pi_ern'][e_low] + ss['pi_ern'][e_mid] * cons_equiv_eU['e_mid'] + ss['pi_ern'][e_high] * cons_equiv_eU['e_high']) / weight_tot)
    
    fig, ((ax1, ax2)) = plt.subplots(1,2)

    ax1.bar(r1, 100 *cons_equiv_eN['e_low'], width = barWidth, label = 'Inc. Perc.: %s' %float("{:.2f}".format(pc1))) 
    ax1.bar(r2, 100 *cons_equiv_eN['e_mid'], width = barWidth, label ='Inc. Perc.: %s' %float("{:.2f}".format(pc2)))
    ax1.bar(r3, 100 *cons_equiv_eN['e_high'], width = barWidth, label ='Inc. Perc.: %s' %float("{:.2f}".format(pc3)), color = 'darkgreen')
    ax1.plot(line_list, np.zeros(11) + eN_avg,  linestyle='--', linewidth=1, color='black', label = 'Average')
    
    ax2.bar(r1, 100 *cons_equiv_eU['e_low'], width = barWidth, label = 'Inc. Perc.: %s' %float("{:.1f}".format(pc1))) 
    ax2.bar(r2, 100 *cons_equiv_eU['e_mid'], width = barWidth, label ='Inc. Perc.: %s' %float("{:.1f}".format(pc2)))
    ax2.bar(r3, 100 *cons_equiv_eU['e_high'], width = barWidth, label ='Inc. Perc.: %s' %float("{:.1f}".format(pc3)), color = 'darkgreen')
    ax2.plot(line_list, np.zeros(11) + eU_avg,  linestyle='--', linewidth=1, color='black', label = 'Average')
    
    ax1.set_xlabel('Wealth Decile')
    ax1.set_ylabel('Pct. of steady-state Consumption')  
    ax2.set_xlabel('Wealth Decile')
    ax2.set_ylabel('Pct. of steady-state Consumption')      
    ax1.set_title('Employed')
    ax2.set_title('Unemployed')
    ax1.plot(line_list, np.zeros(11),  linestyle='-', linewidth=1, color='black')
    ax2.plot(line_list, np.zeros(11),  linestyle='-', linewidth=1, color='black')
    ax2.legend(loc = 'best' )
    ax1.set_xticks(np.arange(len(x_dec)+1))

    ax2.set_xticks(np.arange(len(x_dec)+1))

    ax1.set_xlim(0.5,10.5)
    ax2.set_xlim(0.5,10.5)

    plt.tight_layout()
    plt.gcf().set_size_inches(7, 2.6) 
    plt.rcParams.update({'axes.titlesize': 'x-small'})
    plt.rcParams.update({'axes.labelsize': 'small'})  
    plt.tight_layout()
    plt.show()   

    return fig 
 
@njit
def util(c, eis):
    """Return optimal c, n as function of u'(c) given parameters"""
    utility = c**(1-1/eis)/(1-1/eis) 
    return utility

def C_equiv(ss, css, ctd, Dss, Dtd, index, beta_full):
    (Time,_,_) = Dtd.shape 
    ft = Time 
    time_index = np.arange(Time)
    c_reshape = ctd.reshape(ctd.shape[0], -1)
    beta_full = beta_full.reshape(beta_full.shape[0], -1)
    css = css.flatten()
    Dss = Dss.flatten()
    Dtd = Dtd.reshape(Dtd.shape[0], -1)
    #Atd = Atd.reshape(Atd.shape[0], -1)
    
    ndim = c_reshape.shape[1]
    
    tt = np.reshape(np.broadcast_to(np.arange(ft)[:, np.newaxis], (ft, ndim)), (ft, ndim))    
    
    
    beta = beta_full[:, index]**tt[:, index]
    
    mass = np.sum(Dss[index])
    

    
    U_td_agg = 0 
    hmm = np.empty([ft])
    for k in range(ft):
        #index_td = create_deciles_index_single(Atd[k,:], Dtd[k,:], Atd[k,:].flatten(), dec)
        index_td = index
        c_   = c_reshape[k, index_td]

        beta_td = beta_full[k, index_td]**tt[k, index_td]
       
        U = util(c_, ss['eis'])  * beta_td
        #mass_corr = 1
        mass_corr = mass / np.sum(Dtd[k,index_td].flatten())
      
        U_td_agg += np.vdot(U, Dtd[k,index_td] * mass_corr)
        hmm[k] =  np.vdot(c_, Dtd[k,index_td]* mass_corr)
    pct_dC = 100*(hmm/(np.vdot(Dss[index],css[index]))-1)
    #peak_dC = max(abs(pct_dC))
    
    peak_dC = pct_dC[0]
    peakdC_index =  np.argmax(-pct_dC)
    
    #plt.plot(100*(hmm/(np.vdot(Dss[index],css[index]))-1))
    #plt.show() 
    U_diff = lambda x :  np.vdot(np.sum( (util( (1+x) * css[index], ss['eis'])  )    * beta[np.arange(ft)], axis = 0), Dss[index]) - U_td_agg
    
    cons_equiv = optimize.fsolve(U_diff, 0)
    
    monot_sign = False
    if monot_sign:
        cons_equiv = - abs(cons_equiv)
    
    return cons_equiv, peak_dC, peakdC_index 
   

def Welfare_equiv_by_N(ss, td):
    nPoints = ss['nPoints']
    aN, aS = Utils2.N_specific_var('a', ss)
    DN, DU = Utils2.N_specific_var('D', ss)
    cN, cU = Utils2.N_specific_var('c', ss)
    ft = td['C'].size 
    DNtd, DUtd = Utils2.N_specific_var_td('D', ss, td)
    cNtd, cUtd = Utils2.N_specific_var_td('c', ss, td)    
    DNtd, DUtd = Utils2.N_specific_var_td('D', ss, td)
    
    
    
    deciles_N = Utils2.create_deciles_index(ss['a'], ss['D'], aN.flatten())
    deciles_U = Utils2.create_deciles_index(ss['a'], ss['D'], aS.flatten())
    beta_full = np.reshape(np.broadcast_to(ss['beta'][np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], (ft, nPoints[0], nPoints[1], 2, nPoints[2])), (ft, nPoints[0] * nPoints[1], 2,nPoints[2]))    
    beta_full = beta_full[:,:,0,:]
    
    numer_percs = 10 
    cons_equiv_N = np.empty([numer_percs]) 
    cons_equiv_U = np.empty([numer_percs]) 
    for j in range(numer_percs): 
        cons_equiv_N[j],_,_ = C_equiv(ss, cN, cNtd, DN , DNtd, deciles_N[j], beta_full)
        cons_equiv_U[j],_,_ = C_equiv(ss, cU, cUtd, DU , DUtd, deciles_U[j], beta_full)
            
    return cons_equiv_N, cons_equiv_U


def Welfare_equiv_by_N_Fig(ss, cons_equiv_N, cons_equiv_U, axzero):

    

    x_dec = [*range(1, 11, 1)] 
    barWidth = 0.33
    r1 = [*range(1, 11, 1)] 
    r2 = [x + barWidth for x in r1]
    
    line_list1 = [*range(1, 12, 1)] 
    line_list = [x -0.5 for x in line_list1]
    pylab.rcParams.update(params)
    fig, ((ax1)) = plt.subplots(1,1)
    
    ax1.bar(r1, 100 *cons_equiv_N, width = barWidth, label = 'Employed')
    ax1.bar(r2, 100 *cons_equiv_U, width = barWidth, label = 'Unemployed')
 
    
    ax1.set_xlabel('Wealth Decile')
    ax1.set_ylabel('Pct. of steady-state Consumption')     
    ax1.plot(line_list, np.zeros(11),  linestyle='-', linewidth=1, color='black')
    #ax2.plot(line_list, np.zeros(11),  linestyle='-', linewidth=1, color='black')
    #ax2.legend(loc = 'best' )
    ax1.set_xticks(np.arange(len(x_dec)+1))

    ax1.legend(loc = 'best' , prop = {'size' : 10})
    ax1.set_xlim(0.5,10.5)
    
    min_y_lim = round(min(100*cons_equiv_U)*1.1,2)
    if axzero:
        ax1.set_ylim(min_y_lim,0)
    
    plt.gcf().set_size_inches(1.4*8/2, 1.4*2.6) 
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    #plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
    pylab.rcParams.update(params)
    plt.tight_layout()
    #plt.savefig('plots/C_analysis/dc_by_decile_decomp_by_N.pdf')    
    #plt.show()   


    return fig 





'''Part 2: HA Analysis Tools '''

def C_equiv_by_e_N_calc(e_list, ss, td, dMat, C_equiv):
    e_list_names = ['e_low', 'e_mid', 'e_high']
    cons_equiv_eN = {}
    cons_equiv_eU = {}

    aN, aS = Utils2.N_specific_var('a', ss)
    DN, DU = Utils2.N_specific_var('D', ss)
    cN, cU = Utils2.N_specific_var('c', ss)
    
    dN_factor = (ss['N'] + dMat['N'])/ss['N']
    dU_factor = (1-(ss['N'] + dMat['N']))/(1-ss['N'])
    
    DNtd, DUtd = Utils2.N_specific_var_td('D', ss, td)
    cNtd, cUtd = Utils2.N_specific_var_td('c', ss, td)  
    
    dec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ft = td['C'].size 
    nPoints = ss['nPoints']
    
    for k in range(3):
        e_grid_spec = np.zeros(ss['e_grid'].shape)
        e_grid_spec[e_list[k]] = 1
        e_index_temp = np.kron(np.kron(np.ones(ss['beta'].shape), e_grid_spec), np.ones(2))
        e_index = np.reshape(e_index_temp, (ss['beta'].size * ss['e_grid'].size, 2))
        
        Bool_index =  np.nonzero(e_index[:,0] > 0) 
        
        
        deciles_N = Utils2.create_deciles_index(ss['a'], ss['D'], aN[Bool_index,:].flatten())
        deciles_U = Utils2.create_deciles_index(ss['a'], ss['D'], aS[Bool_index,:].flatten())
        numer_percs = len(deciles_N) 
        cons_equiv_eN[e_list_names[k]] = np.empty([numer_percs]) * np.nan 
        cons_equiv_eU[e_list_names[k]] = np.empty([numer_percs]) * np.nan 
        

        beta1 = ss['beta']
        beta_full = np.reshape(np.broadcast_to(beta1[np.newaxis, :, np.newaxis, np.newaxis], (ft, nPoints[0], 1, nPoints[2])), (ft, nPoints[0] , nPoints[2]))    
    

        for j in range(numer_percs): 
            if deciles_N[j][0].size == 0:
                cons_equiv_eN[e_list_names[k]][j] = 0
            else:
                cons_equiv_eN[e_list_names[k]][j],_,_ = C_equiv(cN[Bool_index,:], cNtd[:,Bool_index,:], DN[Bool_index,:] , DNtd[:,Bool_index,:]* dN_factor[:, np.newaxis, np.newaxis], deciles_N[j], beta_full, dec[j])
            if deciles_U[j][0].size == 0:
                cons_equiv_eU[e_list_names[k]][j] = 0
            else:
                cons_equiv_eU[e_list_names[k]][j],_,_ = C_equiv(cU[Bool_index,:], cUtd[:,Bool_index,:], DU[Bool_index,:] , DUtd[:,Bool_index,:]* dU_factor[:, np.newaxis, np.newaxis], deciles_U[j], beta_full, dec[j])
                
    sign = False 
    if sign:
        for k in range(3):
            for j in range(10):
                if j < 6: 
                    cons_equiv_eN[e_list_names[k]][j] = - abs(cons_equiv_eN[e_list_names[k]][j])
                    cons_equiv_eU[e_list_names[k]][j] = - abs(cons_equiv_eU[e_list_names[k]][j])
    return cons_equiv_eN, cons_equiv_eU



def dC_decomp_by_N(dc_decomped_N, dc_decomped_U, ss):
    
      
    barWidth = 0.25
    x_dec = [*range(1, 11, 1)] 
    
    r1 = np.arange(10) +1 
    r2 = [x + barWidth for x in r1]
    
    sC = np.empty(10) * np.nan
    deciles1 = Utils2.create_deciles_index(ss['a'].flatten(), ss['D'].flatten(), ss['a'])
    i = 0 
    for x in deciles1:
        sC[i] = 100 * np.sum(   ss['c'][x].flatten() * ss['D'][x].flatten()     ) / ss['C']
        i += 1 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2)) = plt.subplots(1,2)
    ax1.set_xticks(np.arange(len(x_dec)+1))
    ax2.set_xticks(np.arange(len(x_dec)+1))
    pylab.rcParams.update(params)
    dC_totN = dc_decomped_N['w'] + dc_decomped_N['ra'] + dc_decomped_N['Tuni'] + dc_decomped_N['Eq']
    dC_totU = -abs(dc_decomped_U['w']) + dc_decomped_U['ra'] + dc_decomped_U['Tuni'] + dc_decomped_U['Eq']
    
    
    ax1.bar(r1, dC_totN, width = barWidth, label = 'Total', color = 'firebrick')        
    bars1 = np.add(dc_decomped_N['w'], dc_decomped_N['Eq']).tolist()
    bars2 = np.add(bars1, dc_decomped_N['ra']).tolist()
    
    p1 = ax1.bar(r2, dc_decomped_N['w'], width = barWidth, label = 'w', color = 'dodgerblue' )
    p1 = ax1.bar(r2, dc_decomped_N['Eq'], bottom=dc_decomped_N['w'], width = barWidth, label = 'Finding rate', color = 'darkgreen')
    p1 = ax1.bar(r2, dc_decomped_N['ra'], bottom=bars1, width = barWidth, label = 'r', color = 'c')
    if 'Tuni' in dc_decomped_N:
        p1 = ax1.bar(r2, dc_decomped_N['Tuni'], bottom=bars2, width = barWidth, label = 'Transfers', color = 'orange')


    #ax1.bar(r2, c_peak_decomped['w'], width = barWidth)     
    
    plt.locator_params(axis='y', nbins=5)
    ax1.set_xlabel('Wealth Decile')
    ax1.set_ylabel('Pct. of steady-state Consumption')     
    ax1.set_title('Employed')

    ax2.set_xlabel('Wealth Decile')
    ax2.set_ylabel('Pct. of steady-state Consumption')     
    ax2.set_title('Unemployed')    


    max_imp = min(dC_totN)
    ax1.set_ylim([round(max_imp)-1,0])
    ax2.set_ylim([round(max_imp)-1,0])

    ax2.bar(r1, dC_totU, width = barWidth, label = 'Total', color = 'firebrick')        
    bars1 = np.add(-abs(dc_decomped_U['w']), dc_decomped_U['Eq']).tolist()
    bars2 = np.add(bars1, dc_decomped_U['ra']).tolist()
    
    p1 = ax2.bar(r2, -abs(dc_decomped_U['w']), width = barWidth, label = 'w', color = 'dodgerblue' )
    p1 = ax2.bar(r2, dc_decomped_U['Eq'], bottom=-abs(dc_decomped_N['w']), width = barWidth, label = 'Finding rate', color = 'darkgreen')
    p1 = ax2.bar(r2, dc_decomped_U['ra'], bottom=bars1, width = barWidth, label = 'r', color = 'c')
    if 'Tuni' in dc_decomped_U:
        p1 = ax2.bar(r2, dc_decomped_U['Tuni'], bottom=bars2, width = barWidth, label = 'Transfers', color = 'orange')

    line_list1 = [*range(1, 11, 1)] 
    line_list = [x  for x in line_list1]
    ax1.plot(line_list, np.zeros(10) + np.average(dC_totN),  linestyle='--', linewidth=1, color='black', label = 'Average')
    ax2.plot(line_list, np.zeros(10) + np.average(dC_totU),  linestyle='--', linewidth=1, color='black', label = 'Average')
    
    ax1.legend(loc='best', ncol=3, prop={'size': 6})
    plt.gcf().set_size_inches(7, 2.6) 
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    #plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
    pylab.rcParams.update(params)
    plt.tight_layout()
    #plt.savefig('plots/C_analysis/dc_by_decile_decomp_by_N.pdf')    
    #plt.show()   

    return fig 


def return_indi_HH_lin(ss, dMat, Time, EGMhousehold):
    tvar_orgshock = {'time' : np.arange(0,Time)}
    
    str_lst = ['w', 'Eq', 'ra']
    xlist   = ['Tuni']
    for k in str_lst:
        if k in dMat:      
                tvar_orgshock[k] = ss[k] + dMat[k]
    for k in xlist:
        if k in dMat:      
            str_lst.append(k)
            tvar_orgshock[k] = ss[k] + dMat[k]
            
            
    td =   EGMhousehold.jac(ss, Time, str_lst)
    td_d = {}
    for key in td:
        td_d[key] = np.empty([Time]) + ss[key]
        for key2 in str_lst:
            td_d[key] =+  td[key][key2] @ dMat[key2]
    return td_d, str_lst, tvar_orgshock


def return_indi_HH_(ss, dMat, Time, EGMhousehold):
    tvar_orgshock = {'time' : np.arange(0,Time)}

    str_lst = ['w', 'Eq', 'N', 'ra']
    xlist   = ['Tuni']
    for k in str_lst:
        if k in dMat:      
                tvar_orgshock[k] = ss[k] + dMat[k]
    for k in xlist:
        if k in dMat:      
            str_lst.append(k)
            tvar_orgshock[k] = ss[k] + 0.9 * dMat[k]
            

    td =   EGMhousehold.td(ss, returnindividual = True, monotonic=False, **tvar_orgshock)

    return td, str_lst, tvar_orgshock

def return_indi_HH(ss, dMat, Time, EGMhousehold):
    tvar_orgshock = {'time' : np.arange(0,Time)}

    str_lst = ['w', 'Eq', 'N', 'ra']
    xlist   = ['Tuni']
    for k in str_lst:
        if k in dMat:      
                tvar_orgshock[k] = ss[k] + dMat[k]
    for k in xlist:
        if k in dMat:      
            str_lst.append(k)
            tvar_orgshock[k] = ss[k] + dMat[k]
            

    td =   EGMhousehold.td(ss, returnindividual = True, monotonic=False, **tvar_orgshock)

    return td, str_lst, tvar_orgshock


def return_indi_HH_specifick_shock(ss, dMat, Time, EGMhousehold, shock_str):
    tvar_orgshock = {'time' : np.arange(0,Time)}
       
    tvar_orgshock[shock_str] = ss[shock_str] + dMat[shock_str]
        

    td =   EGMhousehold.td(ss, returnindividual = True, monotonic=False, **tvar_orgshock)
    #tvar_orgshock['time'] = Time
    return td


def dC_decomp_p0(css, ss, dec, A, Atd, str_lst, deciles_org, tvar_orgshock, dMat, EGMhousehold):
    dc_decomp = {}
    numer_percs = 10 
    peakdC_index = np.empty([numer_percs])  + 0
    Time = dMat['C'].size 
    
    for j in str_lst:
        tvar       = {'time' : np.arange(0,Time), j : tvar_orgshock[j]}
        td       =   EGMhousehold.td(ss,       returnindividual = True, monotonic=False, **tvar)    
        dc_decomp[j] = np.empty([numer_percs])         
        if j == 'N':          
            #dN = ss['N'] + dMat['N']
            Dtd_temp = N_mult('D', ss, td, tvar_orgshock['N'])
        else:
            Dtd_temp = td['D']
        ssD = ss['D'].flatten()
        Dtd_= Dtd_temp.reshape(Dtd_temp.shape[0], -1)
        ctd = td['c'].reshape(td['c'].shape[0], -1)
        css = ss['c'].flatten()                    
                
        for k in range(10):           
            if j == 'N':
                #deciles = create_deciles_index(td['a'][0,:,:], Dtd_temp[0,:,:], td['a'][0,:,:].flatten())
                Dtd = Utils2.N_mult_het(ss, td, tvar_orgshock['N'], dec[k])
                Dtd_= Dtd_temp.reshape(Dtd.shape[0], -1)
            else:
                deciles = deciles_org
            mass = np.sum(ssD[deciles[k]])    

            temp = mass / np.sum(Dtd_[peakdC_index[k].astype(int) , deciles[k]] ) 

            Dtd = Dtd_[peakdC_index[k].astype(int), deciles[k]] * temp
            td_dc = np.vdot(ctd[peakdC_index[k].astype(int), deciles[k]],  Dtd)

            dC = 100*(td_dc/(np.vdot(ssD[deciles[k]], css[deciles[k]]))-1)

            dc_decomp[j][k] = dC
    return dc_decomp



def  C_decomp_peak_change_by_N(css, ss, dec, A, Atd, str_lst, deciles_org, peakdC_index, tvar_orgshock, dMat, EGMhousehold):
    dc_decomped_N = {}
    dc_decomped_U = {}
    numer_percs = 10 
    Time = dMat['C'].size
    
    for j in str_lst:
        if j != 'N':
            tvar       = {'time' : np.arange(0,Time), j : tvar_orgshock[j]}
            td       =   EGMhousehold.td(ss,       returnindividual = True, monotonic=False, **tvar)    
            dc_decomped_N[j] = np.empty([numer_percs]) 
            dc_decomped_U[j] = np.empty([numer_percs]) 
            
            aN, aU = Utils2.N_specific_var('a', ss)
            decilesN = Utils2.create_deciles_index(ss['a'], ss['D'], aN.flatten())
            decilesU = Utils2.create_deciles_index(ss['a'], ss['D'], aU.flatten())
            

            Dtd_tempN, Dtd_tempU  = Utils2.N_specific_var_td('D', ss, td)  
            ssDN, ssDU = Utils2.N_specific_var('D', ss) 
        
            Dtd_N= Dtd_tempN.reshape(Dtd_tempN.shape[0], -1)
            Dtd_U= Dtd_tempU.reshape(Dtd_tempU.shape[0], -1)
            
            ctd_N, ctd_U = Utils2.N_specific_var_td('c', ss, td)  
            ctd_N = ctd_N.reshape(ctd_N.shape[0], -1)
            ctd_U = ctd_U.reshape(ctd_U.shape[0], -1)
            
            cssN, cssU = Utils2.N_specific_var('c', ss) 
            cssN = cssN.flatten()  
            cssU = cssU.flatten()  
            ssDN = ssDN.flatten()  
            ssDU = ssDU.flatten()              
                    
            for k in range(numer_percs):           

                massN = np.sum(ssDN[decilesN[k]])    
                massU = np.sum(ssDU[decilesU[k]])  
                
                tempN = massN / np.sum(Dtd_N[0, decilesN[k]]) 
                tempU = massU / np.sum(Dtd_U[0, decilesU[k]]) 
  
                td_dcN = np.vdot(ctd_N[0, decilesN[k]],  Dtd_N[0,decilesN[k]] * tempN)
                td_dcU = np.vdot(ctd_U[0, decilesU[k]],  Dtd_U[0,decilesU[k]] * tempU)

                dC_N = 100*(td_dcN/(np.vdot(ssDN[decilesN[k]], cssN[decilesN[k]]))-1)
                dC_U = 100*(td_dcU/(np.vdot(ssDU[decilesU[k]], cssU[decilesU[k]]))-1)

                dc_decomped_N[j][k] = dC_N
                dc_decomped_U[j][k] = dC_U

    return dc_decomped_N, dc_decomped_U


def dC_decomp_and_C_deciles(c_peak_decomped, ss):
    
    barWidth = 0.25
    x_dec = [*range(1, 11, 1)] 
    
    r1 = np.arange(10) +1 
    r2 = [x + barWidth for x in r1]
    
    sC = np.empty(10) * np.nan
    deciles1 = Utils2.create_deciles_index(ss['a'].flatten(), ss['D'].flatten(), ss['a'])
    i = 0 
    for x in deciles1:
        sC[i] = 100 * np.sum(   ss['c'][x].flatten() * ss['D'][x].flatten()     ) / ss['C']
        i += 1 
    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2)) = plt.subplots(1,2)
    ax1.set_xticks(np.arange(len(x_dec)+1))
    
    use_sum_as_total = True
    if use_sum_as_total:        
        peak_dC_plot = 0
        for k in c_peak_decomped:
            peak_dC_plot += c_peak_decomped[k]
    else: 
        peak_dC_plot = peak_dC
    
    #c_peak_decomped['N'] = peak_dC_plot - (c_peak_decomped['w'] + c_peak_decomped['q'] +  c_peak_decomped['ra'] + c_peak_decomped['Tuni'])
    
    ax1.bar(r1, peak_dC_plot, width = barWidth, label = 'Total', color = 'firebrick') # hatch='///'     
    
    bars1 = np.add(c_peak_decomped['w'], c_peak_decomped['Eq']).tolist()
    bars2 = np.add(bars1, c_peak_decomped['ra']).tolist()
    bars3 = np.add(bars2, c_peak_decomped['N']).tolist()
    
    p1 = ax1.bar(r2, c_peak_decomped['w'], width = barWidth, label = 'w', color = 'dodgerblue')
    p1 = ax1.bar(r2, c_peak_decomped['Eq'], bottom=c_peak_decomped['w'], width = barWidth, label = 'Finding rate', color = 'darkgreen')
    p1 = ax1.bar(r2, c_peak_decomped['ra'], bottom=bars1, width = barWidth, label = 'r', color = 'c')
    p1 = ax1.bar(r2, c_peak_decomped['N'], bottom=bars2, width = barWidth, label = 'N', color = 'teal')
    if 'Tuni' in c_peak_decomped:
        p1 = ax1.bar(r2, c_peak_decomped['Tuni'], bottom=bars3, width = barWidth, label = 'Transfers', color = 'orange')
    
    #ax1.bar(r2, c_peak_decomped['w'], width = barWidth)     
    
    plt.locator_params(axis='y', nbins=5)
    ax1.set_xlabel('Wealth Decile')
    ax1.set_ylabel('Pct. of steady-state Consumption')  
    ax2.set_xlabel('Wealth Decile')
    ax2.set_ylabel('Share of change in Agg. Consumption')      
    ax1.set_title('On impact Consumption Drop')
    #ax2.set_title('Distribution of Consumption (Steady-state)')
    ax2.set_title('Effect on Aggregate Consumption')
    ax1.legend(loc='best', ncol=3, prop={'size': 7})
    max_imp = min(peak_dC_plot)
    ax1.set_ylim([round(max_imp)-1,0])
    
    print(sum( (peak_dC_plot * sC )/100))
    
    ax2.set_xticks(np.arange(len(x_dec)+1))
    temmp = (peak_dC_plot * sC ) / sum(peak_dC_plot * sC )
    ax2.bar(x_dec, temmp, width = 0.5)
    #ax2.bar(x_dec, sC, width = 0.5)
    
    
    plt.gcf().set_size_inches(7, 2.6) 
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    #plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
    plt.tight_layout()
    
    return fig     





def FigPlot_newss_asset_vars(ss, dMat, Ivar, tplotl, dZ, scen):
    
    # ss - dict 
    # G_jac - jacobian 
    # Ivar - string of shock variable 
    # dZ path of shock variable      
    # tplotl plot time horizon

    
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    
    
    t_start = 50
    dC = 100 * dMat['C'] / ss['C']
    ax1.plot(  dC[t_start:tplotl])
    ax1.plot(np.zeros(tplotl-t_start),  linestyle='--', linewidth=1, color='black')
    ax1.set_title('Consumption')
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct.')

    dA = 100 * dMat['A'] / ss['A']
    ax2.plot(  dC[t_start:tplotl])
    ax2.plot(np.zeros(tplotl-t_start),  linestyle='--', linewidth=1, color='black')
    ax2.set_title('Assets')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct.')
     
    if scen == 'T':
        dT = 100 * dMat['Tuni'] / ss['w']
        ax3.plot(  dT[t_start:tplotl])
        ax3.plot(np.zeros(tplotl-t_start),  linestyle='--', linewidth=1, color='black')
        ax3.set_title('Transfers')
        ax3.set_xlabel('quarters')
        ax3.set_ylabel('Pct. of Wage')
    
    elif scen == 'B':
        dB = 100 * dMat['B'] / ss['B']
        ax3.plot(  dB[t_start:tplotl])
        ax3.plot(np.zeros(tplotl-t_start),  linestyle='--', linewidth=1, color='black')
        ax3.set_title('Bonds')
        ax3.set_xlabel('quarters')
        ax3.set_ylabel('Pct.')     

    plt.gcf().set_size_inches(5*1.6, 2*1.6) 
    plt.rcParams.update({'axes.titlesize': 'x-large'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    fig.tight_layout()

    return fig
    
 
    
'''Search and Matching stuff'''

    
def fig_LM_N_V_q(ss1,ss2,ss3,dMat1,dMat2,dMat3, plot_hori): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard')
    ax1.plot(100 * dMat3['N'][:plot_hori]/ss3['N'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax1.legend() 
    ax1.set_title('Employment')
    
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    
    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard')
    ax2.plot(100 * dMat3['V'][:plot_hori]/ss3['V'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['q'], label = 'Standard')
    ax3.plot(100 * dMat3['q'][:plot_hori]/ss3['q'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 7})
    plot_simple = True
    if plot_simple:        
        ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'Sunk Cost - simple')
        ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'Sunk Cost - simple')       
        ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['q'], label = 'Sunk Cost- simple')
        
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    fig.set_size_inches(7*1.3, 2*1.3) 
    fig.tight_layout()
      
    return fig 
 
def fig_LM_N_V_q_inc_destr(ss1,ss2,ss3,dMat1,dMat2,dMat3, plot_hori): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard')
    ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'Sunk Cost - simple')
    ax1.plot(100 * dMat3['N'][:plot_hori]/ss3['N'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax1.legend() 
    ax1.set_title('Employment')
    
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard')
    ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'Sunk Cost - simple')
    ax2.plot(100 * dMat3['V'][:plot_hori]/ss3['V'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['q'], label = 'Standard')
    ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['q'], label = 'Sunk Cost- simple')
    ax3.plot(100 * dMat3['q'][:plot_hori]/ss3['q'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 8})
 
    ax4.plot(100 * dMat1['destr'][:plot_hori], label = 'Standard')
    ax4.plot(100 * dMat2['destr'][:plot_hori], label = 'Sunk Cost - simple')
    ax4.plot(100 * dMat3['destr'][:plot_hori], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax4.set_title('Seperation Rate')    
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. points Deviation from SS') 
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    fig.set_size_inches(7/1.1,5/1.1 ) 
    fig.tight_layout()
      
    return fig 


     
def fig_LM_N_V_q_destr(ss1,ss2,ss3,dMat1,dMat2,dMat3, dMat4, plot_hori): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard')
    ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'Sunk Cost - simple')
    ax1.plot(100 * dMat3['N'][:plot_hori]/ss3['N'], label = 'Sunk Cost - FR, $d\delta^O$', color = 'Darkgreen')
    ax1.plot(100 * dMat4['N'][:plot_hori]/ss3['N'], label = 'Sunk Cost - FR, $d\delta^{NO}$', color = 'Darkgreen', linestyle = '--')
    
    ax1.legend() 
    ax1.set_title('Employment')
    plt.gcf().set_size_inches(7*1.3, 2*1.3) 
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    
    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard')
    ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'Sunk Cost - simple')
    ax2.plot(100 * dMat3['V'][:plot_hori]/ss3['V'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax2.plot(100 * dMat4['V'][:plot_hori]/ss3['V'], label = 'Sunk Cost - FR', color = 'Darkgreen', linestyle ='--')
    
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['q'], label = 'Standard')
    ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['q'], label = 'Sunk Cost- simple')
    ax3.plot(100 * dMat3['q'][:plot_hori]/ss3['q'], label = 'Sunk Cost - FR', color = 'Darkgreen')
    ax3.plot(100 * dMat4['q'][:plot_hori]/ss3['q'], label = 'Sunk Cost - FR', color = 'Darkgreen', linestyle ='--')
    
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 6})
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    fig.tight_layout()
      
    return fig   
    
      
def fig_LM_N_V_q_destr_shock(ss1,ss2,dMat1,dMat2,dMat3, plot_hori, ss_simple, dMat_simple): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard')
    ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'FR - $d\delta^{NO}$', color = 'Darkgreen')
    ax1.plot(100 * dMat3['N'][:plot_hori]/ss2['N'], label = 'FR - $d\delta^{O}$', color = 'Darkgreen', linestyle ='--')
    ax1.legend() 
    ax1.set_title('Employment')
    
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard')
    ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'FR - $d\delta^{NO}$', color = 'Darkgreen')
    ax2.plot(100 * dMat3['V'][:plot_hori]/ss2['V'], label = 'FR - $d\delta^{O}$', color = 'Darkgreen', linestyle ='--')
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['N'], label = 'Standard')
    ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['N'], label = 'FR - $d\delta^{NO}$', color = 'Darkgreen')
    ax3.plot(100 * dMat3['q'][:plot_hori]/ss2['N'], label = 'FR - $d\delta^{O}$', color = 'Darkgreen', linestyle ='--')
    ax3.set_title('Job-finding rate')
    
    
    
    plot_simple = True
    if plot_simple:
        s=0.7
        ax1.plot(100 * dMat_simple['N'][:plot_hori]*s/ss_simple['N'], label = 'Simple FR', color = 'firebrick', linestyle = 'dotted' )
        ax2.plot(100 * dMat_simple['V'][:plot_hori]*s/ss_simple['V'], label = 'Simple FR', color = 'firebrick', linestyle = 'dotted')
        ax3.plot(100 * dMat_simple['q'][:plot_hori]*s/ss_simple['N'], label = 'Simple FR', color = 'firebrick', linestyle = 'dotted')
            
        
    ax1.legend(loc='best',prop={'size': 8})
 
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    plt.gcf().set_size_inches(7*1.3, 2*1.3) 

    fig.tight_layout()
      
    return fig 




def fig_LM_N_V_q_inc_destr_new_PE(ss1,ss2, dMat1_org, dMat1, dMat2_org, dMat2, plot_hori, ss_simple, dMat_simple_org, dMat_simple): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    lstyle1 = '--'
    lstyle2 = '-'
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard - endo. sep.', linestyle = lstyle1)
    ax1.plot(100 * dMat1_org['N'][:plot_hori]/ss1['N'], label = 'Standard', color = '#E24A33', linestyle = lstyle2)
    ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'FR - endo. sep.', color = 'Darkgreen', linestyle = lstyle1)
    ax1.plot(100 * dMat2_org['N'][:plot_hori]/ss2['N'], label = 'FR', color = 'Darkgreen', linestyle = lstyle2)
    ax1.legend() 
    ax1.set_title('Employment')
    
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')

    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard', linestyle = lstyle1)
    ax2.plot(100 * dMat1_org['V'][:plot_hori]/ss1['V'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax2.plot(100 * dMat2_org['V'][:plot_hori]/ss2['V'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['q'], label = 'Standard', linestyle = lstyle1)
    ax3.plot(100 * dMat1_org['q'][:plot_hori]/ss1['q'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['q'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax3.plot(100 * dMat2_org['q'][:plot_hori]/ss2['q'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 6})
 
    ax4.plot(100 * dMat1['destr'][:plot_hori], label = 'Standard', linestyle = lstyle1)
    ax4.plot(np.zeros(plot_hori), label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax4.plot(100 * dMat2['destr'][:plot_hori], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax4.plot(np.zeros(plot_hori), label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax4.set_title('Seperation Rate')    
    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. points Deviation from SS') 
    
    plot_simple = True 
    if plot_simple:
        lstyle_simple = 'dotted'
        ax1.plot(100 * dMat_simple['N'][:plot_hori]/ss_simple['N'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax1.plot(100 * dMat_simple_org['N'][:plot_hori]/ss_simple['N'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax2.plot(100 * dMat_simple['V'][:plot_hori]/ss_simple['V'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax2.plot(100 * dMat_simple_org['V'][:plot_hori]/ss_simple['V'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax3.plot(100 * dMat_simple['q'][:plot_hori]/ss_simple['q'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax3.plot(100 * dMat_simple_org['q'][:plot_hori]/ss_simple['q'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax4.plot(100 * dMat_simple['destr'][:plot_hori], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax4.plot(np.zeros(plot_hori), label = 'Simple FR', color = 'firebrick', linestyle = '-')
                                                              
    ax1.legend(loc='best',prop={'size': 6})        
        
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')

    
    fig.set_size_inches(7/1.1,5/1.1 ) 

    fig.tight_layout()
      
    return fig 



def fig_LM_N_V_q_inc_destr_new(ss1,ss2, dMat1_org, dMat1, dMat2_org, dMat2, plot_hori, ss_simple, dMat_simple_org, dMat_simple): 

    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    
    lstyle1 = '--'
    lstyle2 = '-'
    
    ax1.plot(100 * dMat1['N'][:plot_hori]/ss1['N'], label = 'Standard - endo. sep.', linestyle = lstyle1)
    ax1.plot(100 * dMat1_org['N'][:plot_hori]/ss1['N'], label = 'Standard', color = '#E24A33', linestyle = lstyle2)
    
    ax1.plot(100 * dMat2['N'][:plot_hori]/ss2['N'], label = 'FR - endo. sep.', color = 'Darkgreen', linestyle = lstyle1)
    ax1.plot(100 * dMat2_org['N'][:plot_hori]/ss2['N'], label = 'FR', color = 'Darkgreen', linestyle = lstyle2)
    
    ax1.set_title('Employment')
    
    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax5.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax6.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    ax2.plot(100 * dMat1['V'][:plot_hori]/ss1['V'], label = 'Standard', linestyle = lstyle1)
    ax2.plot(100 * dMat1_org['V'][:plot_hori]/ss1['V'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax2.plot(100 * dMat2['V'][:plot_hori]/ss2['V'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax2.plot(100 * dMat2_org['V'][:plot_hori]/ss2['V'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat1['q'][:plot_hori]/ss1['q'], label = 'Standard', linestyle = lstyle1)
    ax3.plot(100 * dMat1_org['q'][:plot_hori]/ss1['q'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax3.plot(100 * dMat2['q'][:plot_hori]/ss2['q'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax3.plot(100 * dMat2_org['q'][:plot_hori]/ss2['q'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 6})
 
    ax6.plot(100 * dMat1['destr'][:plot_hori], label = 'Standard', linestyle = lstyle1)
    ax6.plot(np.zeros(plot_hori), label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax6.plot(100 * dMat2['destr'][:plot_hori], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax6.plot(np.zeros(plot_hori), label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax6.set_title('Seperation Rate')    
    ax6.set_xlabel('quarters')
    ax6.set_ylabel('Pct. points Deviation from SS') 
    

    ax4.plot(100 * dMat1['CTD'][:plot_hori]/ss1['CTD'], label = 'Standard', linestyle = lstyle1)
    ax4.plot(100 * dMat1_org['CTD'][:plot_hori]/ss1['CTD'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax4.plot(100 * dMat2['CTD'][:plot_hori]/ss2['CTD'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax4.plot(100 * dMat2_org['CTD'][:plot_hori]/ss2['CTD'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax4.set_title('Consumption')

    ax5.plot(100 * dMat1['Y'][:plot_hori]/ss1['Y'], label = 'Standard', linestyle = lstyle1)
    ax5.plot(100 * dMat1_org['Y'][:plot_hori]/ss1['Y'], label = 'Standard - exo. sep', color = '#E24A33', linestyle = lstyle2)
    ax5.plot(100 * dMat2['Y'][:plot_hori]/ss2['Y'], label = 'FR', color = 'Darkgreen', linestyle = lstyle1)
    ax5.plot(100 * dMat2_org['Y'][:plot_hori]/ss2['Y'], label = 'FR - exo. sep', color = 'Darkgreen', linestyle = lstyle2)
    ax5.set_title('Output')
    
    
    plot_simple = True
    if plot_simple: 
        lstyle_simple = 'dotted'
        ax1.plot(100 * dMat_simple['N'][:plot_hori]/ss_simple['N'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax1.plot(100 * dMat_simple_org['N'][:plot_hori]/ss_simple['N'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax2.plot(100 * dMat_simple['V'][:plot_hori]/ss_simple['V'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax2.plot(100 * dMat_simple_org['V'][:plot_hori]/ss_simple['V'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax3.plot(100 * dMat_simple['q'][:plot_hori]/ss_simple['q'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax3.plot(100 * dMat_simple_org['q'][:plot_hori]/ss_simple['q'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax4.plot(100 * dMat_simple['CTD'][:plot_hori]/ss_simple['CTD'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax4.plot(100 * dMat_simple_org['CTD'][:plot_hori]/ss_simple['CTD'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax5.plot(100 * dMat_simple['Y'][:plot_hori]/ss_simple['Y'], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax5.plot(100 * dMat_simple_org['Y'][:plot_hori]/ss_simple['Y'], label = 'Simple FR', color = 'firebrick', linestyle = '-')
        ax6.plot(100 * dMat_simple['destr'][:plot_hori], label = 'Simple FR - endo. sep.', color = 'firebrick', linestyle = lstyle_simple)
        ax6.plot(np.zeros(plot_hori), label = 'Simple FR', color = 'firebrick', linestyle = '-')
                                                              
    ax1.legend(loc='best',prop={'size': 6})

            
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct. Deviation from SS')
    
    fig.set_size_inches(7*1.3, 4*1.3) 

    fig.tight_layout()
      
    return fig 