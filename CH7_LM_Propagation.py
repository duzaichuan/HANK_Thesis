import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, './Misc')
from Models import *
from LM_funcs import *

import FigUtils
%config InlineBackend.figure_format = 'retina'


#%%

calib = 'Solve'          # Solve with pre-specified values 
settings = {'save' : False, 'use_saved' : True}
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
Time = 300 

dividend, LM = Choose_LM(settings)

Asset_block_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res,  arbitrage, EGMhousehold, MutFund, Fiscal_stab_T],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

prod_stuff = solved(block_list=[monetary1, pricing, ProdFunc, firm_investment2],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  


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
def firm_labor_costly_vac_FR(N, mc, Y, alpha, pMatch, vacK, destr,destrO, w,  L, rstar, r, nPos, Fvac, JV, JM, mu, mux, destrNO, V):   
    MPL = (1-alpha) *  mc * Y / N
    free_entry = JV - nPos * Fvac
    j = N / (N+V)
    JV_res = JV - (- vacK - mux * (1-j) + pMatch * JM +(1-pMatch) * (1-destrO) * JV(+1) /(1+r(+1)))  
    JM_res = JM - ((MPL - w - mux * j - mu) + (1-destrO) * (1-destrNO)/(1+r(+1)) * JM(+1) + destrNO * (1-destrO)  * JV(+1) /(1+r(+1)) )      
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
def endo_destrO(destrO, destrOss, Jxss, rstar, JV, JM, r, destrNO, eps_x, N, V):
    j_p = N/(N+V)
    Jx_p = j_p * (1-destrNO) * JM(+1) + (1-j_p * (1-destrNO)) * JV(+1)
    destrO_Res = destrO - destrOss * ( (1+rstar) / (1+rstar) * Jx_p/Jxss)**(-eps_x)
    mux = destrOss * eps_x / (eps_x-1) * Jxss  / (1+rstar) * (1 - ((1+rstar) / (1+r(+1)) *Jx_p/Jxss)**(1-eps_x))
    return mux, destrO_Res 


def Choose_LM(settings):
    if settings['SAM_model'] == 'Standard':
        dividend = dividend_standard
        LM = solved(block_list=[laborMarket1, laborMarket2, firm_labor_standard, wage_func1, destr_rate],  
                    unknowns=['N', 'S', 'Tight',  'JV', 'JM'],
                    targets=[ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res'] )  
        if settings['endo_destrNO'] :
            LM = solved(block_list=[laborMarket1, laborMarket2, firm_labor_standard, wage_func1, destr_rate, endo_destr],  
                        unknowns=['N', 'S', 'Tight',  'JV', 'JM', 'destrNO'],
                        targets=[ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )              
            
    elif settings['SAM_model'] == 'Costly_vac_creation':
        dividend = dividend_costly_vac
        if settings['SAM_model_variant'] == 'simple':    
            LM = solved(block_list=[laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wage_func1, destr_rate],  
                        unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM'],
                        targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res'] )  
            if settings['endo_destrNO']:
                LM = solved(block_list=[laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wage_func1, destr_rate, endo_destr],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrNO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )                  
        elif settings['SAM_model_variant'] == 'FR': 
            if settings['endo_destrNO']:
                LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, destr_rate, endo_destr],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrNO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrNO_Res'] )  
            elif settings['endo_destrO']:
                LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, destr_rate, endo_destrO],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM', 'destrO'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res', 'destrO_Res'] )   
            else:
                LM = solved(block_list=[laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1],  
                            unknowns=['N', 'S', 'Tight', 'V', 'JV', 'JM'],
                            targets=[ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res'] )       
    return dividend, LM 
         


#%% PE comparison of sunk cost models 

IvarPE = 'Z'
IvarGE = 'mup'
GE_shock = 0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

ss_standard, ss_costly, ss_costly_FR, G_jac_standard, G_jac_costly_vac,  G_jac_costly_vac_FR, _ = LM_models_shock(IvarPE)


dMat_standard = FigUtils.Shock_mult(G_jac_standard, dZ_PE, IvarPE)
dMat_costly_vac = FigUtils.Shock_mult(G_jac_costly_vac, dZ_PE, IvarPE)
dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)

plot_hori = 30 
fig = FigUtils.fig_LM_N_V_q(ss_standard, ss_costly, ss_costly_FR, dMat_standard, dMat_costly_vac, dMat_costly_vac_FR, 30)
#plt.savefig('plots/Labor_market/model_comp.pdf')

plt.show() 


# Importance of wages vs MPL
n_pars = 10
shares = np.linspace(0,0.3,n_pars)

from cycler import cycler

def vac2wages_sensetivity():

    exogenous = [IvarPE]
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plot_hori = 30 
    new_colors = [plt.get_cmap('copper')(1. * i/n_pars) for i in range(n_pars)]
    
    #ax1.set_prop_cycle( cmap='pink')
    #plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    ax1.set_prop_cycle(cycler('color', new_colors))
    ax2.set_prop_cycle(cycler('color', new_colors))
    ax3.set_prop_cycle(cycler('color', new_colors))

    settings['Fvac_share'] = False
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'     
    j = 0 
    for x in shares:   
        settings['vac2w_costs'] = x
        ss_costly_FR = ss_calib('Solve', settings)
        
        block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate]
        unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y']
        targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res']   
            
        G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
        dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)
        if j == 0 or j == n_pars-1:
            #lab = r'$\frac{(\kappa_x x_t^2 +\kappa_v)/m_t}{w}$=' + f"{settings['vac2w_costs']:.1f}"
            lab = r'$\frac{MPL}{w}$=' + f"{1+settings['vac2w_costs']:.1f}"            
        else:
            lab = None 
        ax1.plot(100 * dMat_costly_vac_FR['N'][:plot_hori]/ss_costly_FR['N'], alpha = 1, label = lab)
        ax2.plot(100 * dMat_costly_vac_FR['V'][:plot_hori]/ss_costly_FR['V'], alpha = 1)
        ax3.plot(100 * dMat_costly_vac_FR['q'][:plot_hori]/ss_costly_FR['q'], alpha = 1)
        j += 1 
    
    
    ax1.legend(loc='best',prop={'size': 8})
    plt.gcf().set_size_inches(7*1.3, 2*1.3) 
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    ax1.set_title('Employment')
    ax2.set_title('Vacancies')
    ax3.set_title('Job-finding rate')
    fig.tight_layout()
    return fig 

fig = vac2wages_sensetivity()


# what does fixed cost vs recurring cost mean for FR model?
 
n_pars = 10
shares = np.linspace(0,1,n_pars)

from cycler import cycler

def Fvac_sensetivity():

    exogenous = [IvarPE]
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plot_hori = 30 
    new_colors = [plt.get_cmap('copper')(1. * i/n_pars) for i in range(n_pars)]
    
    #ax1.set_prop_cycle( cmap='pink')
    #plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    ax1.set_prop_cycle(cycler('color', new_colors))
    ax2.set_prop_cycle(cycler('color', new_colors))
    ax3.set_prop_cycle(cycler('color', new_colors))
    settings['vac2w_costs'] = 0.05 
    
    j = 0 
    for x in shares:   
        settings['Fvac_share'] = True 
        settings['Fvac_factor'] = x
        settings['SAM_model'] = 'Costly_vac_creation'
        settings['SAM_model_variant'] = 'FR'
        ss_costly_FR = ss_calib('Solve', settings)
        
        block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate]
        unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y']
        targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res']   
            
        G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
        dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)
        if j == 0:
            lab = r'$\kappa_x$=' + f"{ss_costly_FR['Fvac']:.2f}" + r', $\kappa_V$=' + f"{ss_costly_FR['vacK']:.2f}"
        elif j == n_pars-1: 
            lab = r'$\kappa_x$=' + f"{ss_costly_FR['Fvac']:.2f}" + r', $\kappa_V$=' + f"{ss_costly_FR['vacK']:.2f}"
        else:
            lab = None 
        ax1.plot(100 * dMat_costly_vac_FR['N'][:plot_hori]/ss_costly_FR['N'], alpha = 1, label = lab)
        ax2.plot(100 * dMat_costly_vac_FR['V'][:plot_hori]/ss_costly_FR['V'], alpha = 1)
        ax3.plot(100 * dMat_costly_vac_FR['q'][:plot_hori]/ss_costly_FR['q'], alpha = 1)
        j += 1 
    
    
    ax1.legend(loc='best',prop={'size': 8})
    plt.gcf().set_size_inches(7*1.3, 2*1.3) 
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    ax1.set_title('Employment')
    ax2.set_title('Vacancies')
    ax3.set_title('Job-finding rate')
    fig.tight_layout()
    return fig 

fig = Fvac_sensetivity()

# Finding rates from different models -> households 
settings['Fvac_share'] = False 
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05 
        
settings['SAM_model'] = 'Standard'
ss_standard = ss_calib('Solve', settings)   
  
settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'simple'
ss_costly = ss_calib('Solve', settings)

settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'FR'
ss_costly_FR = ss_calib('Solve', settings)

ttt = np.arange(0,Time)

tvar = {'time' : ttt ,'Eq' : ss_standard['q'] + dMat_standard['q']} 
td_q_standard =   EGMhousehold.td(ss_standard, returnindividual = False, monotonic=False, **tvar)
tvar = {'time' : ttt ,'Eq' : ss_costly['q'] + dMat_costly_vac['q']} 
td_q_costly_vac =   EGMhousehold.td(ss_costly, returnindividual = False, monotonic=False, **tvar)
tvar = {'time' : ttt ,'Eq' : ss_costly_FR['q'] + dMat_costly_vac_FR['q']} 
td_q_costly_vac_FR =   EGMhousehold.td(ss_costly_FR, returnindividual = False, monotonic=False, **tvar)

dC_standard = (td_q_standard['C']/ss_standard['C']-1)*100
dC_costly_vac = (td_q_costly_vac['C']/ss_costly['C']-1)*100
dC_costly_vac_FR = (td_q_costly_vac_FR['C']/ss_costly_FR['C']-1)*100

dA_standard = (td_q_standard['A']/ss_standard['A']-1)*100
dA_costly_vac = (td_q_costly_vac['A']/ss_costly['A']-1)*100
dA_costly_vac_FR = (td_q_costly_vac_FR['A']/ss_costly_FR['A']-1)*100

pylab.rcParams.update(params)
fig, ((ax1,ax2)) = plt.subplots(1, 2)

plot_hori = 30   

ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')

ax1.plot(dC_standard[:plot_hori], label = 'Standard')
ax1.plot(dC_costly_vac[:plot_hori], label = 'Sunk Cost - simple')
ax1.plot(dC_costly_vac_FR[:plot_hori], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Deviation from SS')
ax1.legend() 
ax1.set_title('Consumption')
ax2.set_title('Assets')

ax2.plot(dA_standard[:plot_hori], label = 'Standard')
ax2.plot(dA_costly_vac[:plot_hori], label = 'Sunk Cost - simple')
ax2.plot(dA_costly_vac_FR[:plot_hori], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Deviation from SS')
plt.gcf().set_size_inches(5*1.6, 2*1.6) 
#plt.savefig('plots/Labor_market/dC_dq_partial.pdf')

fig.tight_layout()

#%% Sunk costs in GE

calibration = 'Solve'

settings['endo_destrNO'] = False
settings['endo_destrO'] = False
settings['Fvac_share'] = False 
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05
settings['destrO_share'] = 0.5

exogenous = [IvarGE]  
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']

settings['SAM_model'] = 'Standard'
ss_standard = ss_calib(calibration, settings)   
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_standard, save=False)
dMat_standard = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'simple'
ss_costly = ss_calib(calibration, settings)
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly, save=False)
dMat_costly_vac = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)


settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'FR'
ss_costly_FR = ss_calib(calibration, settings)
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)


pylab.rcParams.update(params)
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
plot_hori = 60 

ax1.plot(100 * dMat_standard['N'][:plot_hori]/ss['N'], label = 'Standard')
ax1.plot(100 * dMat_costly_vac['N'][:plot_hori]/ss_costly['N'], label = 'Sunk Cost - simple')
ax1.plot(100 * dMat_costly_vac_FR['N'][:plot_hori]/ss_costly_FR['N'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax1.legend() 
ax1.set_title('Employment')


ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax5.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax6.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')

ax2.plot(100 * dMat_standard['V'][:plot_hori]/ss_standard['V'], label = 'Standard')
ax2.plot(100 * dMat_costly_vac['V'][:plot_hori]/ss_costly['V'], label = 'Sunk Cost - simple')
ax2.plot(100 * dMat_costly_vac_FR['V'][:plot_hori]/ss_costly_FR['V'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax2.set_title('Vacancies')

ax3.plot(100 * dMat_standard['q'][:plot_hori]/ss_standard['q'], label = 'Standard')
ax3.plot(100 * dMat_costly_vac['q'][:plot_hori]/ss_costly['q'], label = 'Sunk Cost- simple')
ax3.plot(100 * dMat_costly_vac_FR['q'][:plot_hori]/ss_costly_FR['q'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax3.set_title('Job-finding rate')

ax4.plot(100 * dMat_standard['CTD'][:plot_hori]/ss_standard['CTD'], label = 'Standard')
ax4.plot(100 * dMat_costly_vac['CTD'][:plot_hori]/ss_costly['CTD'], label = 'Sunk Cost- simple')
ax4.plot(100 * dMat_costly_vac_FR['CTD'][:plot_hori]/ss_costly_FR['CTD'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax4.set_title('Consumption')

ax5.plot(100 * dMat_standard['Y'][:plot_hori]/ss_standard['Y'], label = 'Standard')
ax5.plot(100 * dMat_costly_vac['Y'][:plot_hori]/ss_costly['Y'], label = 'Sunk Cost- simple')
ax5.plot(100 * dMat_costly_vac_FR['Y'][:plot_hori]/ss_costly_FR['Y'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax5.set_title('Output')

ax6.plot(100 * dMat_standard['I'][:plot_hori]/ss_standard['I'], label = 'Standard')
ax6.plot(100 * dMat_costly_vac['I'][:plot_hori]/ss_costly['I'], label = 'Sunk Cost- simple')
ax6.plot(100 * dMat_costly_vac_FR['I'][:plot_hori]/ss_costly_FR['I'], label = 'Sunk Cost - FR', color = 'Darkgreen')
ax6.set_title('Investment')

fig.set_size_inches(7*1.3, 4*1.3) 


ax1.legend(loc='best',prop={'size': 6})

ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Deviation from SS')
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Deviation from SS')
ax3.set_xlabel('quarters')
ax3.set_ylabel('Pct. Deviation from SS')
ax4.set_xlabel('quarters')
ax4.set_ylabel('Pct. Deviation from SS')
ax5.set_xlabel('quarters')
ax5.set_ylabel('Pct. Deviation from SS')
ax6.set_xlabel('quarters')
ax6.set_ylabel('Pct. Deviation from SS')

fig.tight_layout()



#%% Job destruction rate shock 

ss_standard, ss_costly, ss_costly_FR, G_jac_standard, G_jac_costly_vac, G_jac_costly_vac_FR_destrO, G_jac_costly_vac_FR_NO = LM_models_shock('destr')

rhos = 10
ddestr = np.zeros([Time])
ddestr[:rhos] =  0.01  

dZ_FR_destrO = np.zeros([Time])
dZ_FR_destrO[:rhos] = 0.01 / (1 - ss_costly_FR['destrNO'])  # destrO
dZ_FR_destrNO = np.zeros([Time])
dZ_FR_destrNO[:rhos] = 0.01/ (1 - ss_costly_FR['destrO'])      # destrNO

dMat_standard = FigUtils.Shock_mult(G_jac_standard, ddestr, 'destr')
dMat_costly_vac = FigUtils.Shock_mult(G_jac_costly_vac, ddestr, 'destr')
dMat_costly_vac_FR_destrO  = FigUtils.Shock_mult(G_jac_costly_vac_FR_destrO, dZ_FR_destrO, 'destrO')
dMat_costly_vac_FR_destrNO = FigUtils.Shock_mult(G_jac_costly_vac_FR_NO, dZ_FR_destrNO, 'destrNO')


fig = FigUtils.fig_LM_N_V_q_destr_shock(ss_standard, ss_costly_FR, dMat_standard, dMat_costly_vac_FR_destrNO, dMat_costly_vac_FR_destrO, 30, ss_costly, dMat_costly_vac)



#%% PE with endo. Normal separation rate 

def LM_models_shock_endo_sep(Ivar):  
    exogenous = [Ivar]
    settings = {'save' : False, 'use_saved' : True, 'Fvac_share' : False}
    settings['Fvac_factor'] = 5
    settings['vac2w_costs'] = 0.05    
    settings['destrO_share'] = 0.5 
    
    settings['SAM_model'] = 'Standard'
    ss_standard = ss_calib('Solve', settings)     
    ss_standard.update({ 'eps_m' : 0.5})
    
    block_list=[laborMarket1, laborMarket2, firm_labor_standard, wage_func1, ProdFunc, destr_rate, endo_destr, destr_rate_lag]
    unknowns = ['N', 'S', 'Tight',  'JV', 'JM', 'Y', 'destrNO']
    targets = [ 'N_res', 'S_res',  'free_entry', 'JV_res', 'JM_res', 'ProdFunc_Res', 'destrNO_Res']
    
    G_jac_standard = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_standard, save=False)
    
    
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'simple'
    ss_costly = ss_calib('Solve', settings)
    ss_costly.update({ 'eps_m' : 0.5})
    
    block_list = [laborMarket1_costly_vac, laborMarket2, firm_labor_costly_vac, wage_func1, ProdFunc, destr_rate, endo_destr, destr_rate_lag]
    unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y', 'destrNO']
    targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res', 'destrNO_Res']   
        
    G_jac_costly_vac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly, save=False)
    
    # FR 
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    ss_costly_FR = ss_calib('Solve', settings)
    ss_costly_FR.update({ 'eps_m' : 0.5})
    
    block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate, endo_destr, destr_rate_lag]
    unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y', 'destrNO']
    targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res', 'destrNO_Res']   
        
    G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)

    return ss_standard, ss_costly, ss_costly_FR, G_jac_standard, G_jac_costly_vac,  G_jac_costly_vac_FR


ss_standard, ss_costly, ss_costly_FR, G_jac_standard, G_jac_costly_vac,  G_jac_costly_vac_FR= LM_models_shock_endo_sep(IvarPE)


dMat_standard = FigUtils.Shock_mult(G_jac_standard, dZ_PE, IvarPE)
dMat_costly_vac = FigUtils.Shock_mult(G_jac_costly_vac, dZ_PE, IvarPE)
dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)

# exo sep 
ss_standard, _, ss_costly_FR, G_jac_standard, G_jac_costly_vac,  G_jac_costly_vac_FR, _ = LM_models_shock(IvarPE)
dMat_standard_exo_sep = FigUtils.Shock_mult(G_jac_standard, dZ_PE, IvarPE)
dMat_costly_vac_FR_exo_sep = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)
dMat_costly_vac_exo_sep = FigUtils.Shock_mult(G_jac_costly_vac, dZ_PE, IvarPE)

plot_hori = 30 
#fig = FigUtils.fig_LM_N_V_q_inc_destr(ss_standard, ss_costly, ss_costly_FR, dMat_standard, dMat_costly_vac, dMat_costly_vac_FR, plot_hori, plot_simple)
fig = FigUtils.fig_LM_N_V_q_inc_destr_new_PE(ss_standard, ss_costly_FR, dMat_standard_exo_sep, dMat_standard, dMat_costly_vac_FR_exo_sep, dMat_costly_vac_FR, plot_hori, ss_costly, dMat_costly_vac_exo_sep, dMat_costly_vac)
#plt.savefig('plots/Labor_market/model_comp_endo_destr_partial_eq.pdf')
plt.show() 


#%% What does seperation elasciticity imply for impulses (PE)

n_pars = 10
shares = np.linspace(0,2,n_pars)

from cycler import cycler

def eps_m_sensetivity():
    
    Ivar = 'Z'
    exogenous = [Ivar]
    pylab.rcParams.update(params)
    
    figS, (ax1S) = plt.subplots(1,1)
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
    
    plot_hori = 30 
    new_colors = [plt.get_cmap('copper')(1. * i/n_pars) for i in range(n_pars)]
    
    #ax1.set_prop_cycle( cmap='pink')
    #plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    ax1.set_prop_cycle(cycler('color', new_colors))
    ax2.set_prop_cycle(cycler('color', new_colors))
    ax3.set_prop_cycle(cycler('color', new_colors))
    ax4.set_prop_cycle(cycler('color', new_colors))
    ax1S.set_prop_cycle(cycler('color', new_colors))
    settings = {'save' : False, 'use_saved' : True, 'Fvac_share' : False}
    settings['Fvac_factor'] = 5
    settings['vac2w_costs'] = 0.05     
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    ss_costly_FR = ss_calib('Solve', settings)
    j = 0 
    for x in shares:   
        if np.isclose(x,1):
            x = 0.99
        ss_costly_FR.update({ 'eps_m' : x})
        
        block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate, endo_destr]
        unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y', 'destrNO']
        targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res', 'destrNO_Res'] 
                
        G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
        dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)
        if j == 0:
            lab = r'$\epsilon_{m}$=' + str(x)
        elif j == n_pars-1: 
            lab = r'$\epsilon_{m}$=' + str(x)
        else:
            lab = None 
        ax1.plot(100 * dMat_costly_vac_FR['N'][:plot_hori]/ss_costly_FR['N'], alpha = 1, label = lab)
        ax2.plot(100 * dMat_costly_vac_FR['V'][:plot_hori]/ss_costly_FR['V'], alpha = 1)
        ax3.plot(100 * dMat_costly_vac_FR['q'][:plot_hori]/ss_costly_FR['q'], alpha = 1)
        ax4.plot(100 * dMat_costly_vac_FR['destr'][:plot_hori], alpha = 1)
        ax1S.plot(100 * dMat_costly_vac_FR['S'][:plot_hori]/ss_costly_FR['S'], alpha = 1, label = lab)
        
        j += 1 
        
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')    
    ax1S.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')    

    ax1S.set_xlabel('quarters')
    ax1S.set_ylabel('Pct. Deviation from SS')
    figS.set_size_inches(7/1.5,5/1.5 ) 
    ax1S.legend(loc='best',prop={'size': 8})
    
    ax1.legend(loc='best',prop={'size': 8})
    fig.set_size_inches(7/1.1,5/1.1 ) 
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    ax1.set_title('Employment')
    ax2.set_title('Vacancies')
    ax3.set_title('Job-finding rate')
    ax4.set_title('Seperation Rate')  
    ax4.set_ylabel('Pct. points Deviation from SS') 
    ax4.set_xlabel('quarters')
    fig.tight_layout()    
    
    return fig

fig = eps_m_sensetivity()

#%% General eq. with endo destrNO for different models 

 
# Calculate Jacobian 
exogenous = [IvarGE]  
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']


settings['endo_destrNO'] = True
settings['endo_destrO'] = False
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05 
#calibration = 'Partial_calib'
calibration = 'Solve'

settings['SAM_model'] = 'Standard'
ss_standard = ss_calib(calibration, settings)   
ss_standard.update({ 'eps_m' : 0.5})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_standard, save=False)
dMat_standard = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['endo_destrNO'] = False
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_standard, save=False)
dMat_standard_org = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'simple'
settings['endo_destrNO'] = False

ss_costly = ss_calib(calibration, settings)
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly, save=False)
dMat_costly_vac_org = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['endo_destrNO'] = True
ss_costly.update({ 'eps_m' : 0.15})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly, save=False)
dMat_costly_vac = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['endo_destrNO'] = True
settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'FR'
ss_costly_FR = ss_calib(calibration, settings)
ss_costly_FR.update({ 'eps_m' : 0.5})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

settings['endo_destrNO'] = False
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, destr_rate_lag, Eq] 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
dMat_costly_vac_FR_org = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)


plot_hori = 60 
fig = FigUtils.fig_LM_N_V_q_inc_destr_new(ss_standard, ss_costly_FR, dMat_standard_org, dMat_standard, dMat_costly_vac_FR_org, dMat_costly_vac_FR, plot_hori, ss_costly, dMat_costly_vac_org, dMat_costly_vac)


#%% Endo. destrO  in FR PE

n_pars = 10
shares = np.linspace(0,2,n_pars)

from cycler import cycler

def epsV_sensetivity():
    
    Ivar = 'Z'
    exogenous = [Ivar]
    pylab.rcParams.update(params)
    figS, (ax1S) = plt.subplots(1,1)
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
    

    plot_hori = 30 
    new_colors = [plt.get_cmap('copper')(1. * i/n_pars) for i in range(n_pars)]
    
    #ax1.set_prop_cycle( cmap='pink')
    #plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    ax1.set_prop_cycle(cycler('color', new_colors))
    ax2.set_prop_cycle(cycler('color', new_colors))
    ax3.set_prop_cycle(cycler('color', new_colors))
    ax4.set_prop_cycle(cycler('color', new_colors))
    ax1S.set_prop_cycle(cycler('color', new_colors))
    
    settings['Fvac_share'] = False 
    settings['Fvac_factor'] = 5
    settings['vac2w_costs'] = 0.05         
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    
    ss_costly_FR = ss_calib('Solve', settings)
    j = 0 
    for x in shares:   
        if np.isclose(x,1):
            x = 0.99
     
        ss_costly_FR.update({ 'eps_x' : x})
        
        block_list = [laborMarket1_costly_vac_FR, laborMarket2, firm_labor_costly_vac_FR, wage_func1, ProdFunc, destr_rate, endo_destrO]
        unknowns   = ['N', 'S', 'Tight', 'V', 'JV', 'JM', 'Y', 'destrO']
        targets    = [ 'N_res', 'S_res', 'Match_res', 'free_entry', 'JV_res', 'JM_res',  'ProdFunc_Res', 'destrO_Res'] 
                
        G_jac_costly_vac_FR = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR, save=False)
        dMat_costly_vac_FR = FigUtils.Shock_mult(G_jac_costly_vac_FR, dZ_PE, IvarPE)
        if j == 0:
            lab = r'$\epsilon_{x}$=' + str(x)
        elif j == n_pars-1: 
            lab = r'$\epsilon_{x}$=' + str(x)
        else:
            lab = None       
        ax1.plot(100 * dMat_costly_vac_FR['N'][:plot_hori]/ss_costly_FR['N'], alpha = 1, label = lab)
        ax2.plot(100 * dMat_costly_vac_FR['V'][:plot_hori]/ss_costly_FR['V'], alpha = 1)
        ax3.plot(100 * dMat_costly_vac_FR['q'][:plot_hori]/ss_costly_FR['q'], alpha = 1)
        ax4.plot(100 * dMat_costly_vac_FR['destr'][:plot_hori], alpha = 1)
        ax1S.plot(100 * dMat_costly_vac_FR['S'][:plot_hori]/ss_costly_FR['S'], alpha = 1)
        
        j += 1 
        
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')    
    ax1S.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')   


    ax1S.set_ylabel('Pct. Deviation from SS') 
    ax1S.set_xlabel('quarters')
    ax1S.legend(loc='best',prop={'size': 6})
    figS.tight_layout()
    figS.set_size_inches(7/1.1,5/1.1 ) 
    #plt.savefig('plots/Labor_market/different_eps_x_searchers.pdf')
    
    ax1.legend(loc='best',prop={'size': 8})
    fig.set_size_inches(7/1.1,5/1.1 ) 
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')
    ax1.set_title('Employment')
    ax2.set_title('Vacancies')
    ax3.set_title('Job-finding rate')
    ax4.set_title('Seperation Rate')  
    ax4.set_ylabel('Pct. points Deviation from SS') 
    ax4.set_xlabel('quarters')
    fig.tight_layout()
    #plt.savefig('plots/Labor_market/different_eps_x.pdf')

    return fig, figS

fig, _ = epsV_sensetivity()


#%% Endo. detrO in General eq. 

exogenous = [IvarGE]  
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']

settings['Fvac_share'] = False 
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05  

settings['q_calib'] = False
settings['destrO_share'] = 0.5

# Baseline
settings['SAM_model'] = 'Standard'
settings['endo_destrNO'] = False
settings['endo_destrO'] = False
ss = ss_calib('Solve', settings)
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=False)
dMat_standard = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

# FR 
settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'FR'


# endo destrO
settings['endo_destrNO'] = False
settings['endo_destrO'] = True
ss_costly_FR_endoO = ss_calib('Solve', settings)
eps_x = 0.5
ss_costly_FR_endoO.update({'eps_x' : eps_x})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endoO, save=False)
dMat_costly_vac_FR_endoO = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)


# # endo destrNO
settings['endo_destrNO'] = True 
settings['endo_destrO'] = False
ss_costly_FR_endoNO = ss_calib('Solve', settings)
eps_m = eps_x
ss_costly_FR_endoNO.update({'eps_m' : eps_m})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endoNO, save=False)
dMat_costly_vac_FR_endoNO = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)


# exo sep
settings['endo_destrNO'] = False
settings['endo_destrO'] = False 

ss_costly_FR_base = ss_calib('Solve', settings)

dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 

G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_base, save=False)
dMat_costly_vac_FR_base = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

#%%

pylab.rcParams.update(params)
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
plot_hori = 60 

plot_destrNO = False

ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax5.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
ax6.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')

lstyle1 = '--'
lstyle2 = '-'
col2    = 'firebrick'

ax1.plot(100 * dMat_standard['N'][:plot_hori]/ss['N'], label = 'Standard')
ax1.plot(100 * dMat_costly_vac_FR_base['N'][:plot_hori]/ss_costly_FR_base['N'], label = 'FR - exo. sep.', color = 'Darkgreen')
ax1.plot(100 * dMat_costly_vac_FR_endoO['N'][:plot_hori]/ss_costly_FR_endoO['N'], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)

#ax1.legend() 
ax1.set_title('Employment')


ax2.plot(100 * dMat_standard['V'][:plot_hori]/ss['V'], label = 'Standard')
ax2.plot(100 * dMat_costly_vac_FR_base['V'][:plot_hori]/ss_costly_FR_base['V'], label = 'FR - exo. sep.', color = 'Darkgreen')
ax2.plot(100 * dMat_costly_vac_FR_endoO['V'][:plot_hori]/ss_costly_FR_endoO['V'], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)

ax2.set_title('Vacancies')


ax3.plot(100 * dMat_standard['q'][:plot_hori]/ss['q'], label = 'Standard')
ax3.plot(100 * dMat_costly_vac_FR_base['q'][:plot_hori]/ss_costly_FR_base['q'], label = 'FR - exo. sep.', color = 'Darkgreen')
ax3.plot(100 * dMat_costly_vac_FR_endoO['q'][:plot_hori]/ss_costly_FR_endoO['q'], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)
ax3.set_title('Job-finding rate')

ax4.plot(100 * dMat_standard['CTD'][:plot_hori]/ss['CTD'], label = 'Standard')
ax4.plot(100 * dMat_costly_vac_FR_base['CTD'][:plot_hori]/ss_costly_FR_base['CTD'], label = 'FR - exo. sep.', color = 'Darkgreen')
ax4.plot(100 * dMat_costly_vac_FR_endoO['CTD'][:plot_hori]/ss_costly_FR_endoO['CTD'], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)
ax4.set_title('Consumption')

ax5.plot(100 * dMat_standard['Y'][:plot_hori]/ss['Y'], label = 'Standard')
ax5.plot(100 * dMat_costly_vac_FR_base['Y'][:plot_hori]/ss_costly_FR_base['Y'], label = 'FR - exo. sep.', color = 'Darkgreen')
ax5.plot(100 * dMat_costly_vac_FR_endoO['Y'][:plot_hori]/ss_costly_FR_endoO['Y'], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)
ax5.set_title('Output')

ax6.plot(np.zeros(plot_hori), label = 'Standard')
ax6.plot(np.zeros(plot_hori), label = 'FR - exo. sep.', color = 'Darkgreen')
ax6.plot(100 * dMat_costly_vac_FR_endoO['destr'][:plot_hori], label = 'FR - endo. $\delta^{O}$', color = col2, linestyle = lstyle2)
ax6.set_title('Seperation Rate')    
ax6.set_xlabel('quarters')
ax6.set_ylabel('Pct. points Deviation from SS') 

fig.set_size_inches(7*1.3, 4*1.3) 

if plot_destrNO:
    ax6.plot(100 * dMat_costly_vac_FR_endoNO['destr'][:plot_hori], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)
    ax5.plot(100 * dMat_costly_vac_FR_endoNO['Y'][:plot_hori]/ss_costly_FR_endoNO['Y'], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)
    ax4.plot(100 * dMat_costly_vac_FR_endoNO['C_agg'][:plot_hori]/ss_costly_FR_endoNO['C_agg'], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)
    ax3.plot(100 * dMat_costly_vac_FR_endoNO['q'][:plot_hori]/ss_costly_FR_endoNO['q'], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)
    ax2.plot(100 * dMat_costly_vac_FR_endoNO['V'][:plot_hori]/ss_costly_FR_endoNO['V'], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)
    ax1.plot(100 * dMat_costly_vac_FR_endoNO['N'][:plot_hori]/ss_costly_FR_endoNO['N'], label = 'FR - endo. $\delta^{NO}$', color = 'Darkgreen', linestyle = lstyle1)

ax1.legend(loc='best',prop={'size': 8})

ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Deviation from SS')
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Deviation from SS')
ax3.set_xlabel('quarters')
ax3.set_ylabel('Pct. Deviation from SS')
ax4.set_xlabel('quarters')
ax4.set_ylabel('Pct. Deviation from SS')
ax5.set_xlabel('quarters')
ax5.set_ylabel('Pct. Deviation from SS')

fig.tight_layout()


#%% Compare different model (Full model, exo. sep, no precautionary savings etc.)

def comp_LM_models_GE(IvarGE, dZ_GE):
    exogenous = [IvarGE]  
    unknowns = ['L', 'mc']
    targets = ['Asset_mkt', 'Labor_mkt']
    settings['Fvac_share'] = False 
    settings['Fvac_factor'] = 5
    settings['vac2w_costs'] = 0.05      
    settings['q_calib'] = False
    settings['destrO_share'] = 0.5    
    eps_x = 0.5
    
    Asset_block = Asset_block_T

    # Full FR model with endo. sep delta^O
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
    
    
    ss_costly_FR_endo_destr = ss_calib('Solve', settings)
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    ss_costly_FR_endo_destr.update({'eps_x' : eps_x})
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endo_destr, save=False)
    
    dMat_costly_vac_FR_endo_destr = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    # Full FR model with exo sep.
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = False
  
    ss_costly_FR_exo_destr = ss_calib('Solve', settings)
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_exo_destr, save=False)
    
    dMat_costly_vac_FR_exo_destr = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    
    # Full FR model with endo sep. no precautionary savings
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True


    ss_costly_FR_endo_destr_no_precaut = ss_calib('Solve', settings)
    ss_costly_FR_endo_destr.update({'eps_x' : eps_x})
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend] 
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endo_destr_no_precaut, save=False)    
    dMat_costly_vac_FR_endo_destr_no_precaut = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    # Full FR model with endo sep. no sunk costs 
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
    
    settings['Fvac_share'] = True 
    settings['Fvac_factor'] = 0
    
    ss_costly_FR_endo_destr_no_sunkcost = ss_calib('Solve', settings)
    ss_costly_FR_endo_destr_no_sunkcost.update({'eps_x' : eps_x})
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    ss_costly_FR_endo_destr_no_sunkcost.update({'eps_x' : eps_x})
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endo_destr_no_sunkcost, save=False)
    
    dMat_costly_vac_FR_endo_destr_no_sunkcost = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)

    
    # baseline - no sunk costs or endo. seperations
    settings['SAM_model'] = 'Standard'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = False
    settings['Fvac_share'] = False 
    
    ss_base = ss_calib('Solve', settings)
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_base, save=False)
    
    dMat_base = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
        
    return ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ss_base, dMat_base




def comp_LM_models_GE_figs(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ssbase, dBase):
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    plot_hori = 60 
    
    pl = 1.5
    lstyle_no_risk = 'dashdot'
    marker_no_risk = ''
    markersize_no_risk = 1
    
    ax1.plot(100 * dMat_costly_vac_FR_endo_destr['N'][:plot_hori]/ss_costly_FR_endo_destr['N'], linewidth=pl, label = 'Full Model')
    ax1.plot(100 * dMat_costly_vac_FR_exo_destr['N'][:plot_hori]/ss_costly_FR_exo_destr['N'], linewidth=pl, label = 'Exo. Seperations', linestyle = '--')
    ax1.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['N'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['N'], linewidth=pl, label = 'No unemployment risk', color = 'Darkgreen', linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax1.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['N'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['N'], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    

    
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    
    ax2.plot(100 * dMat_costly_vac_FR_endo_destr['V'][:plot_hori]/ss_costly_FR_endo_destr['V'], linewidth=pl)
    ax2.plot(100 * dMat_costly_vac_FR_exo_destr['V'][:plot_hori]/ss_costly_FR_exo_destr['V'], linewidth=pl, linestyle = '--')
    ax2.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['V'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['V'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax2.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['V'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['V'], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    ax2.set_title('Vacancies')
    
    ax3.plot(100 * dMat_costly_vac_FR_endo_destr['q'][:plot_hori]/ss_costly_FR_endo_destr['q'], linewidth=pl)
    ax3.plot(100 * dMat_costly_vac_FR_exo_destr['q'][:plot_hori]/ss_costly_FR_exo_destr['q'], linewidth=pl, linestyle = '--')
    ax3.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['q'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['q'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax3.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['q'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['q'], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    ax3.set_title('Job-finding rate')
    ax1.legend(loc='best',prop={'size': 6})
    
    ax1.set_xlabel('quarters')
    ax1.set_ylabel('Pct. Deviation from SS')
    ax2.set_xlabel('quarters')
    ax2.set_ylabel('Pct. Deviation from SS')
    ax3.set_xlabel('quarters')
    ax3.set_ylabel('Pct. Deviation from SS')


    ax4.set_xlabel('quarters')
    ax4.set_ylabel('Pct. Deviation from SS')
    ax5.set_xlabel('quarters')
    ax5.set_ylabel('Pct. Deviation from SS')
    ax6.set_xlabel('quarters')
    ax6.set_ylabel('Pct. points Deviation from SS')
    ax6.set_title('Separation Rate')
    
    ax4.set_title('Consumption')
    ax5.set_title('Output')
    
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax5.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax6.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    ax4.plot(100 * dMat_costly_vac_FR_endo_destr['CTD'][:plot_hori]/ss_costly_FR_endo_destr['C'], linewidth=pl)
    ax4.plot(100 * dMat_costly_vac_FR_exo_destr['CTD'][:plot_hori]/ss_costly_FR_exo_destr['C'], linewidth=pl, linestyle = '--')
    ax4.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['CTD'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['C'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax4.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['CTD'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['C'], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    ax5.plot(100 * dMat_costly_vac_FR_endo_destr['Y'][:plot_hori]/ss_costly_FR_endo_destr['Y'], linewidth=pl)
    ax5.plot(100 * dMat_costly_vac_FR_exo_destr['Y'][:plot_hori]/ss_costly_FR_exo_destr['Y'], linewidth=pl, linestyle = '--')
    ax5.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['Y'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['Y'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax5.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['Y'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['Y'], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    ax6.plot(100 * dMat_costly_vac_FR_endo_destr['destr'][:plot_hori], linewidth=pl)
    ax6.plot(np.zeros(plot_hori), linewidth=pl, linestyle = '--')
    ax6.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['destr'][:plot_hori], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_no_risk, marker = marker_no_risk, markersize = markersize_no_risk)
    ax6.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['destr'][:plot_hori], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    plot_base = False
    
    if plot_base:
        markerstyle = 'dashdot'
        col_base = 'black'
        markersize_set = 1
        ax1.plot(100 * dBase['N'][:plot_hori]/ssbase['N'], linewidth=pl, label = 'Basic HANK', linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax2.plot(100 * dBase['V'][:plot_hori]/ssbase['V'], linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax3.plot(100 * dBase['q'][:plot_hori]/ssbase['q'], color = col_base, linewidth=pl, linestyle = markerstyle, markersize = markersize_set)
        ax4.plot(100 * dBase['C_agg'][:plot_hori]/ssbase['C'], linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax5.plot(100 * dBase['Y'][:plot_hori]/ssbase['Y'], linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax6.plot(np.zeros(plot_hori), linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)

    ax1.legend(loc='best',prop={'size': 6})
    ax1.set_title('Employment')
    
    plt.gcf().set_size_inches(8, 5) 
    fig.tight_layout()
    
    return fig 

IvarPE = 'Z'
IvarGE = 'mup'
GE_shock = 0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ss_base, dMat_base = comp_LM_models_GE(IvarGE, dZ_GE)
fig = comp_LM_models_GE_figs(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost,ss_base, dMat_base)



