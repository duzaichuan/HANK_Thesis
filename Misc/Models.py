



import numpy as np
from numba import vectorize, njit, jit, prange, guvectorize, cuda  


import jacobian as jac
import nonlinear

import utils
from het_block import het
from simple_block import simple
from solved_block import solved

import FigUtils


#%%

def New_LR_SS(ss, dZ, Ivar, Time, scenario, t_terminal, HH):


    @solved(unknowns=['K', 'I'], targets=['inv', 'K_res'])
    def firm_investment(K,  alpha, rstar, delta, kappak, Q, mc, Y, I_s, I, r):       
        rk_plus =  alpha * mc(+1) * Y(+1) / K 
        inv = 1 - (rk_plus +  (1-delta))/(1+r(+1))   
        K_res = K - ((1 - delta) * K(-1) + I) 
        return inv, K_res


    # @solved(unknowns=['K', 'Q', 'I'], targets=['inv', 'val', 'K_res'])
    # def firm_investment(K,  alpha, rstar, delta, kappak, Q, mc, Y, I_s, I, r):       
    #     rk_plus =  alpha * mc(+1) * Y(+1) / K 
    #     inv = Q - (rk_plus + Q(+1) * (1-delta))/(1+r(+1))
    #     LHS = 1 + kappak/2 * (I/I(-1) -1)**2   + I/I(-1) * kappak * (I/I(-1) -1) 
    #     RHS = Q(+1)  + kappak * (I(+1)/I -1) * (I(+1)/I)**2   
    #     val = LHS - RHS     
    #     K_res = K - ((1 - delta) * K(-1) + I(-1)) 
    #     return inv, val, K_res
    @simple
    def firm_labor_standard(mc, Y, L, alpha, pMatch, vacK, destr, w, rstar, r, Z, JV, JM):   
        MPL = (1-alpha)  * mc * Y / L   
        free_entry = JV - 0
        JV_res = JV - (- vacK + pMatch * JM)
        JM_res = JM - ((MPL - w) + JM(+1) * (1-destr)/(1+r(+1)))         
        return  free_entry, JV_res, JM_res 
    
    
    @simple
    def ProdFunc(Y, Z, K, alpha, L): 
        ProdFunc_Res = Y - Z * K(-1)**alpha * L**(1-alpha)
        return ProdFunc_Res
    
        
    
    @solved(unknowns=['w'], targets=['w_res'])
    def wage_func(Tight, w, wss, Tightss, pi): 
        eta =  0.01
        w_res = np.log(w) - (np.log(wss) + eta * np.log(Tight/Tightss) )    
        return w_res
    

    @simple
    def laborMarket1(q, N, destr, S, pMatch, Tight):
        N_res = N - ((1-destr) * N(-1) + S * q)    
        S_res = S - (1 - (1-destr) * N(-1))
        V = q * S / pMatch
        N_ = N(-1)
        return N_res, S_res, V, N_
    
    @simple
    def laborMarket2(Tight, ma): 
        q      = Tight / ((1+Tight**ma)**(1/ma))
        pMatch = q / Tight
        return q, pMatch
    
    
    @simple 
    def MutFund(B, r, ra, div,  p):
        MF_Div =  (div + p)  + B(-1) * (1 + r) 
        MF_Div_res = 1+ra - ( MF_Div)
        return  MF_Div_res, MF_Div
    
    @solved(unknowns=['p'], targets=['equity'])
    def arbitrage(div, p, r):
        equity = div(+1) + p(+1) - p * (1 + r(+1))
        return equity
        
    @simple
    def dividend(Y, w, N,vacK, V, I, F_cost, T_firms):
        div = Y - w * N  -  vacK  - I   - F_cost - T_firms 
        return  div
        
    
    @simple 
    def fiscal_rev(C, VAT, B, taxes,  MF_Div, T_firms):  
        G_rev = taxes  + B  + T_firms 
        return G_rev
    
    @simple 
    def fiscal_exp(b, r, G, B, N, lumpsum_T, uT, UINCAGG_count): 
        G_exp = G + UINCAGG_count  + (lumpsum_T + uT)  + B(-1) * (1+r) 
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
    def Asset_mkt_clearing(A_agg, B, p):  
        Asset_mkt = B + p - A_agg 
        return Asset_mkt    
    
    @simple
    def Labor_mkt_clearing(N, L):     
        Labor_mkt = L  -  N
        return Labor_mkt    

    @solved(unknowns=[ 'i',  'r'], targets=['i_res',  'fisher'])
    def monetary(rstar, pi, i, r, phi):
        phi = 1.3
        i_res = i -  (rstar + phi * pi)        
        fisher = 1 + i(-1) - (1 + r) * (1 + pi)    
        return i_res, fisher 

    @simple
    def aggregate(CN, CS, AN, AS, N, ssN, TAXN, TAXS, UINCAGG, TINC, CTD, ATD):
        dN =  N/ssN
        dU = (1-N)/(1-ssN) 
        C_agg =  CTD  
        A_agg = ATD     
        taxes = TINC
        UINCAGG_count  =  UINCAGG * dU
        return C_agg, A_agg, taxes, UINCAGG_count
    
    
    @simple
    def goods_mkt_clearing(Y, C_agg, C, I, G, psip, V , vacK, Isip, F_cost, A_DEBT, kappa):      
        goods_mkt = Y - C_agg  - I  - G  -  vacK   - F_cost + kappa * A_DEBT(-1)
        return goods_mkt   



    LM = solved(block_list=[laborMarket1, laborMarket2, firm_labor_standard, wage_func],  
                unknowns=['N', 'S', 'Tight',  'JV', 'JM'],
                targets=[ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res'] )  
            
    
    Asset_block_B = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, HH, MutFund,  aggregate],
                    unknowns=[ 'B',  'ra'],
                    targets=[  'B_res','MF_Div_res'] )  
    
    Asset_block_only_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, HH,  MutFund,  aggregate, HHTransfer],
                    unknowns=[ 'uT',  'ra'],
                    targets=[  'B_res','MF_Div_res'] )    

    prod_stuff = solved(block_list=[monetary, ProdFunc, firm_investment],
                    unknowns=[ 'Y'],
                    targets=[  'ProdFunc_Res' ] )  

    exogenous = [Ivar]  
    unknowns = [ 'r']
    targets = ['Asset_mkt']
    # general equilibrium jacobians
    if scenario == 'T':
        block_list = [Asset_block_only_T,  Asset_mkt_clearing, dividend, arbitrage, goods_mkt_clearing] 
    if scenario == 'B':
        block_list = [LM, Asset_block_B, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, arbitrage, goods_mkt_clearing] 
        

    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=True)
    
    dMat = FigUtils.Shock_mult(G_jac, dZ, Ivar)

    

    New_ss_temp = {}
    for i in dMat.keys():
        if i in ss:
            New_ss_temp[i] = dMat[i][t_terminal] + ss[i]
    
    New_ss_temp[Ivar] = ss[Ivar] + dZ[t_terminal]
    
    
    New_ss =  ss.copy()
    for key in New_ss_temp.keys():
        New_ss[key] = New_ss_temp[key]  
    del New_ss_temp

    fig = FigUtils.FigPlot_newss_asset_vars(ss, dMat, Ivar, t_terminal, dZ, scenario)  
    #fig = FigUtils.FigPlot3x3(ss, dMat, Ivar, t_terminal, dZ)

    return New_ss,  fig


