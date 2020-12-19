
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from HANKSAM import *
import FigUtils
#from HANKSAM import ss_calib
%config InlineBackend.figure_format = 'retina'


#%% Solve steady state 

calib = 'Solve'          # Solve with pre-specified values 
settings = {'save' : True, 'use_saved' : True}
settings['Fvac_share'] = False 
settings['SAM_model'] = 'Standard'
settings['SAM_model_variant'] = 'simple'

settings['endo_destrNO'] = False
settings['endo_destrO'] = False

# some calibration for labor market 
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05 

# solve the steady state 
ss = ss_calib(calib, settings)     

#%%

# load new steady states (file for producing these to be added)
with open('misc/New_ss_T.pkl', 'rb') as f:
    New_ss_T = pickle.load(f)
with open('misc/New_ss_B.pkl', 'rb') as f:
    New_ss_B = pickle.load(f)
    
#%%
dividend, LM = Choose_LM(settings)

Asset_block_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res,  arbitrage, EGMhousehold, MutFund, Fiscal_stab_T, aggregate],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

prod_stuff = solved(block_list=[monetary, pricing, ProdFunc, firm_investment],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  

# markup shock 
Time = 300   
Ivar = 'mup'
rhos = 0.6
dZ =  0.1 *ss[Ivar] * rhos**(np.arange(Time))

exogenous = [Ivar]  
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, destr_rate_lag, dividend, Eq] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=False)
G_jac_lower_b_T = jac.get_G(block_list, exogenous, unknowns, targets,  Time, New_ss_T, save=False)
G_jac_lower_b_B = jac.get_G(block_list, exogenous, unknowns, targets,  Time, New_ss_B, save=False)

rhos = 0.6
dZ =  0.1 *ss[Ivar] * rhos**(np.arange(Time))

dMat         = FigUtils.Shock_mult(G_jac, dZ, Ivar)
dMat_lower_b_T = FigUtils.Shock_mult(G_jac_lower_b_T, dZ, Ivar)
dMat_lower_b_B = FigUtils.Shock_mult(G_jac_lower_b_B, dZ, Ivar)

# Revenue-neutral 
desc = ['Baseline', 'Lower benefits']
shock_title = 'Productivity'
fig = FigUtils.FigPlot3x3_compare(ss, dMat, New_ss_T, dMat_lower_b_T, Ivar, 30, dZ, desc, shock_title, 'C_agg')
plt.show()

# Not Revenue-neutral 
desc = ['Baseline', 'Lower benefits']
shock_title = 'Productivity'
fig = FigUtils.FigPlot3x3_compare(ss, dMat, New_ss_B, dMat_lower_b_B, Ivar, 30, dZ, desc, shock_title, 'C_agg')
plt.show()


# decompose Consumption  
fig = FigUtils.C_decomp_compare_lin(ss, New_ss_T, dMat, dMat_lower_b_T, 60, Time, EGMhousehold, 'CTD')
fig = FigUtils.C_decomp_compare_lin(ss, New_ss_B, dMat, dMat_lower_b_B, 60, Time, EGMhousehold, 'CTD')

    
#%% welfare


# transtiional dynamics 
ft = Time 
ttt = np.arange(0,Time)

td, str_lst, tvar_orgshock = FigUtils.return_indi_HH_lin(ss, dMat, Time, EGMhousehold)

D      = ss['D']
A_dist = ss['a']
nPoints = ss['nPoints']

beta_full = np.reshape(np.broadcast_to(ss['beta'][np.newaxis, :, np.newaxis, np.newaxis], (ft, nPoints[0], nPoints[1], nPoints[2])), (ft, nPoints[0] * nPoints[1], nPoints[2]))       
tt = np.reshape(np.broadcast_to(np.arange(ft)[:, np.newaxis, np.newaxis, np.newaxis], (ft, nPoints[0], nPoints[1], nPoints[2])), (ft, nPoints[0] * nPoints[1], nPoints[2]))    
dN = ss['N'] + G_jac['N'][Ivar] @ dZ
Dtd = N_mult('D', ss, td, dN)


Dss = ss['D']
c_trans = td['c']
css    = ss['c']
Atd = td['a']
N = ss['N']

deciles = create_deciles_index(ss['a'], ss['D'], ss['a'].flatten())
numer_percs = len(deciles) 
beta_full = np.reshape(np.broadcast_to(ss['beta'][np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], (ft, nPoints[0], nPoints[1], 2, nPoints[2])), (ft, nPoints[0] * nPoints[1], 2, nPoints[2]))    

cons_equiv_agg = np.empty([numer_percs]) 
peak_dC  = np.empty([numer_percs]) 
peakdC_index = np.empty([numer_percs])  
dec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for j in range(numer_percs): 
    cons_equiv_agg[j], peak_dC[j], peakdC_index[j] = FigUtils.C_equiv(ss, css, c_trans, Dss, Dtd, deciles[j], beta_full)
    
#%%

# Welfare by Employment 
ft = Time 
ttt = np.arange(0,Time)
td, str_lst, tvar_orgshock = FigUtils.return_indi_HH(ss, dMat, Time, EGMhousehold)
cons_equiv_N, cons_equiv_U = FigUtils.Welfare_equiv_by_N(ss, td)

#%% Ensure that we are in steady state before using non-linear solver for households 
hhtemp = EGMhousehold.ss(**New_ss_T)
New_ss_T_nl = New_ss_T.copy()
for key in hhtemp:
    New_ss_T_nl[key] = hhtemp[key]
    
hhtemp = EGMhousehold.ss(**New_ss_B)
New_ss_B_nl = New_ss_B.copy()
for key in hhtemp:
    New_ss_B_nl[key] = hhtemp[key]   


#% T
axzero = False
td_lower_b_T, _,_ = FigUtils.return_indi_HH_(New_ss_T_nl, dMat_lower_b_T, Time, EGMhousehold)
cons_equiv_N_lower_b_T, cons_equiv_U_lower_b_T = FigUtils.Welfare_equiv_by_N(New_ss_T_nl, td_lower_b_T)

diff_N = cons_equiv_N_lower_b_T - cons_equiv_N
diff_U = cons_equiv_U_lower_b_T - cons_equiv_U

axzero = False
fig = FigUtils.Welfare_equiv_by_N_Fig(ss, diff_N, diff_U, axzero)

# B
axzero = False
td_lower_b_B, _,_ = FigUtils.return_indi_HH_(New_ss_B_nl, dMat_lower_b_B, Time, EGMhousehold)
cons_equiv_N_lower_b_B, cons_equiv_U_lower_b_B = FigUtils.Welfare_equiv_by_N(New_ss_B_nl, td_lower_b_B)

diff_N = cons_equiv_N_lower_b_B - cons_equiv_N
diff_U = cons_equiv_U_lower_b_B - cons_equiv_U

axzero = False
fig = FigUtils.Welfare_equiv_by_N_Fig(ss, diff_N, diff_U, axzero)




