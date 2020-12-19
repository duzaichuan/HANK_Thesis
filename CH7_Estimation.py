import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, './Misc')
from Models import *
from LM_funcs import *

import FigUtils
%config InlineBackend.figure_format = 'retina'

import estimation as est

#%%

# load data
npzfile = np.load('Employment_data/U_V_data.npz') 
GDP_data = npzfile['A']
U_rate_data = npzfile['B']
V_rate_data = npzfile['C']


# de-mean
U_rate_data -= np.average(U_rate_data)
V_rate_data -= np.average(V_rate_data)
GDP_data -= np.average(GDP_data)

Y = np.empty([len(V_rate_data), 3])
Y[:,0] = GDP_data
Y[:,1] = U_rate_data
Y[:,2] = V_rate_data

# Model  
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']
settings = {'save' : False, 'use_saved' : True}
settings['Fvac_share'] = False 
settings['Fvac_factor'] = 5
settings['vac2w_costs'] = 0.05      
settings['q_calib'] = False
settings['destrO_share'] = 0.5    

prod_stuff = solved(block_list=[monetary1, pricing, ProdFunc, firm_investment2],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  

Asset_block_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res,  arbitrage, EGMhousehold, MutFund, Fiscal_stab_T],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

@simple
def moments(N, V, Y):
    U_rate = 100 * (1 - N)
    V_rate = 100 * V / (N+V)
    return U_rate, V_rate

# Full FR model with endo. sep delta^O
settings['SAM_model'] = 'Costly_vac_creation'
settings['SAM_model_variant'] = 'FR'
settings['endo_destrNO'] = False
settings['endo_destrO'] = True


ss_est = ss_calib('Solve', settings)
ss = ss_est.copy() 
ss_est.update({'eps_r': 0})
dividend, LM = Choose_LM(settings)
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag, moments] 

  

    #%%
    var1 = 'mup'
    var2 = 'Z'
    var3 = 'beta_shock'
    Time = 300 
    exogenous = [var1, var2, var3]
    G = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_est, save=False)
    

    rho_mup_g = 0.9
    rho_r_g   = 0.9
    rho_beta_g   = 0.9
    sigma_mup_g = 0.05
    sigma_r_g = 0.05
    sigma_beta_g = 0.05

    dZ1 = rho_mup_g**(np.arange(Time))
    dY1, dU1, dV1 = G['Y'][var1] @ dZ1, G['U_rate'][var1] @ dZ1, G['V_rate'][var1] @ dZ1
    dX1 = np.stack([ dY1, dU1, dV1], axis=1)
    
    dZ2 = rho_r_g**(np.arange(Time))
    dY2, dU2, dV2 = G['Y'][var2] @ dZ2, G['U_rate'][var2] @ dZ2, G['V_rate'][var2] @ dZ2
    dX2 = np.stack([ dY2, dU2, dV2], axis=1)    

    dZ3 = rho_beta_g**(np.arange(Time))
    dY3, dU3, dV3 = G['Y'][var3] @ dZ3, G['U_rate'][var3] @ dZ3, G['V_rate'][var3] @ dZ3
    dX3 = np.stack([ dY3, dU3, dV3], axis=1)    

    dZ = [dZ1, dZ2, dZ3]
    
    dX = np.stack([dX1, dX2, dX3], axis=2)
    sigmas = np.array([sigma_mup_g, sigma_r_g, sigma_beta_g])
    Sigma = est.all_covariances(dX, sigmas) # burn-in for jit

    loglik = est.log_likelihood(Y, Sigma)


#%%

    def log_likelihood_from_parameters(rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, Y, G, dZ):
        # impulse response to persistent shock
        dZ1, dZ2, dZ3 = dZ
        
        dZ1 = rho_mup**(np.arange(Time))
        dY1, dU1, dV1 = G['Y'][var1] @ dZ1, G['U_rate'][var1] @ dZ1, G['V_rate'][var1] @ dZ1
        dX1 = np.stack([ dY1, dU1, dV1], axis=1)
        
        dZ2 = rho_r**(np.arange(Time))
        dY2, dU2, dV2 = G['Y'][var2] @ dZ2, G['U_rate'][var2] @ dZ2, G['V_rate'][var2] @ dZ2
        dX2 = np.stack([ dY2, dU2, dV2], axis=1)    
    
        dZ3 = rho_beta**(np.arange(Time))
        dY3, dU3, dV3 = G['Y'][var3] @ dZ3, G['U_rate'][var3] @ dZ3, G['V_rate'][var3] @ dZ3
        dX3 = np.stack([ dY3, dU3, dV3], axis=1)    
   
        dX = np.stack([dX1, dX2, dX3], axis=2)
        sigmas = np.array([sigma_mup, sigma_r, sigma_beta])
        Sigma = est.all_covariances(dX, sigmas) # burn-in for jit
    
        # calculate log=likelihood from this
        return est.log_likelihood(Y, Sigma)
    
    
    # minimizer       
    def loglik_obj(x, *args):       
        pen = 0 
        for k in range(len(x)):
            if x[k] < 0:
                x[k] = 0.00001
                pen += 0.1 
        rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta  = x
        rhos = [rho_mup, rho_r, rho_beta]
        for k in range(len(rhos)):
            if rhos[k] > 0.990:
                rhos[k] = 0.990
                pen += 1 * rhos[k]
                
        rho_mup = rhos[0]
        rho_r = rhos[1]     
        rho_beta = rhos[2] 
        
        args_dict = args[0]
        Y = args_dict['Y']
        G = args_dict['G']
        dZ = args_dict['dZ']
        
        #print(rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta)
        loglik = log_likelihood_from_parameters(rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, Y, G, dZ)

        print(loglik)
        obj = -loglik + abs(loglik) * pen # we minimize         
        
        return obj  
    ss_est_copy = ss_est.copy() 
    args = {'Y' : Y, 'G' : G, 'dZ' : dZ, 'ss_est_copy' : ss_est_copy}
    
    result = optimize.minimize(loglik_obj, np.array([rho_mup_g, rho_r_g, rho_beta_g, sigma_mup_g, sigma_r_g, sigma_beta_g]), method='Nelder-Mead',   args=args, tol = 1e-5)  
    rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta = result.x
    
    
#%% estimate paramters also

    def Vacancy_costs_calib(x, *args):
            Fvac_g, vacK_g = x
            (Fvac_factor,) = args
     
            Fvac_res = Fvac_factor - Fvac_g * ss_est_copy['nPos']  / vacK_g
            JV = Fvac_g * ss_est_copy['nPos']
            JM  = (JV - (1-ss_est_copy['pMatch']) *(1-ss_est_copy['destrO']) * JV /(1+ss_est_copy['r']) + vacK_g) / ss_est_copy['pMatch']                                 
            LS_res = JM - ((ss_est_copy['MPL']-ss_est_copy['w']) + (1-ss_est_copy['destrO']) * (1-ss_est_copy['destrNO'])/(1+ss_est_copy['r']) * JM  + ss_est_copy['destrNO'] * (1-ss_est_copy['destrO'])  * JV /(1+ss_est_copy['r']))   
                
            return  np.array([Fvac_res, LS_res]) 
    

    G = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_est, save=True)

    # minimizer       
    def loglik_obj_params(x, *args):       
        args_dict = args[0]
        ss_est_copy = args_dict['ss_est_copy']
        
        pen = 0 
        for k in range(len(x)):
            if x[k] < 0:
                x[k] = 0.00001
                pen += 0.1 
                
        rho1, rho2, rho3, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac_factor  = x

        rhos = [rho1, rho2, rho3]
        for k in range(len(rhos)):
            if rhos[k] > 0.990:
                pen += 1 * rhos[k]
                rhos[k] = 0.990
                
        rho1 = rhos[0]
        rho2 = rhos[1]    
        rho3 = rhos[2] 
                
        if eps_x < 0:
            eps_x = 0 

        Fvac_g = ss_est_copy['Fvac']
        vacK_g = ss_est_copy['vacK']
        
        sol = optimize.root(Vacancy_costs_calib, np.array([Fvac_g, vacK_g]),  method='hybr', args = (Fvac_factor,))
        (Fvac, vacK) = sol.x 
        if not sol.success:
            raise Exception("Solver did not succeed") 
            
             
        JV = Fvac * ss_est_copy['nPos']            
        Vac_costs = vacK  + Fvac * ss_est_copy['nPos'] **2
        JM  = (JV - (1-ss_est_copy['pMatch']) *(1-ss_est_copy['destrO']) * JV /(1+ss_est_copy['r']) + vacK) / ss_est_copy['pMatch']
                   

        if Fvac < 0:
            Fvac = 0    
            pen += 0.3 * rhos[k]
            JV = ss_est['nPos'] * Fvac  
            vacK = -(JV - (1-ss_est_copy['pMatch']) *(1-ss_est_copy['destrO']) * JV /(1+ss_est_copy['r']) - ss_est_copy['JM'] * ss_est_copy['pMatch']     )
        if vacK < 0:
            vacK = 0.001    
            pen += 0.3 * rhos[k]
                
        #JM =  ((ss_est['MPL']-ss_est['w'])  + ss_est['destrNO'] * (1-ss_est['destrO'])  * JV /(1+ss_est['r'])) / (1-(1-ss_est['destrO']) * (1-ss_est['destrNO'])/(1+ss_est['r']))                                       

        ss_est_copy.update({'eps_x' : eps_x, 'Fvac' : Fvac, 'JV' : JV, 'JM' : JM, 'vacK' : vacK})
        G = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_est_copy, save=False, use_saved=True)
               
        Y = args_dict['Y']
        dZ = args_dict['dZ']
        
        #print(rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac)
        print(x)        
        try:
            loglik = log_likelihood_from_parameters(rho1, rho2, rho3, sigma_mup, sigma_r, sigma_beta, Y, G, dZ)
        except np.linalg.LinAlgError:
            loglik = -5e+5

        N, nvar =  Y.shape
        likelihood_con = -  nvar * N / 2 * np.log(2*np.pi)
        print('Likelihood', loglik + likelihood_con)
        
        obj = -loglik + abs(loglik) * pen   # we minimize         
        #obj = -loglik   
        return obj  

    eps_x_g = 1.63394371
    Fvac_factor_g = 3.0398687
    
    # JV = ss_est['nPos'] * Fvac_g  
    # vacK = -(JV - (1-ss_est['pMatch']) *(1-ss_est['destrO']) * JV /(1+ss_est['r']) - ss_est['JM'] * ss_est['pMatch']     )
    # Fvac_factor  = Fvac_g * (ss_est_copy['nPos'] / vacK)
    

#%% Estimate
    rho_mup_g = 0.98812272
    rho_r_g   = 0.54465474
    rho_beta_g = 0.47286501
    sigma_mup_g = 0.01174432
    sigma_r_g   =  0.00649838
    sigma_beta_g =  0.06215774
    
    guess = np.array([rho_mup_g,rho_r_g, rho_beta_g, sigma_mup_g, sigma_r_g, sigma_beta_g, eps_x_g, Fvac_factor_g])
    #guess = np.array([rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x_g, Fvac_factor_g])
    
    import time

    estimate = False
    
    if estimate:
        start = time.time() 
        result1 = optimize.minimize(loglik_obj_params, guess, method='Nelder-mead',   args=args, options = {'xatol' : 1e-04, 'fatol' : 1e-04})  
        end = time.time()
        print(end - start)  
    
        rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac_factor = result1.x
        estimates = result1.x
    else:
        estimates = guess 
        rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac_factor = estimates
        
        Fvac_g = ss_est_copy['Fvac']
        vacK_g = ss_est_copy['vacK']        
        sol = optimize.root(Vacancy_costs_calib, np.array([Fvac_g, vacK_g]),  method='hybr', args = (Fvac_factor,))
        (Fvac, vacK) = sol.x 
        if not sol.success:
            raise Exception("Solver did not succeed") 
                         
        JV = Fvac * ss_est_copy['nPos']            
        Vac_costs = vacK  + Fvac * ss_est_copy['nPos'] **2
        JM  = (JV - (1-ss_est_copy['pMatch']) *(1-ss_est_copy['destrO']) * JV /(1+ss_est_copy['r']) + vacK) / ss_est_copy['pMatch']
 
    
#%% grid search first

    n = 10 
    rel = 0.2
    Fvac_list = np.linspace(Fvac_factor*(1-rel), Fvac_factor*(1+rel), n)
    eps_x_list = np.linspace(eps_x*(1-rel), eps_x*(1+rel), n)
    
    loglike = np.zeros([n,n])
    for k in range(n):
        for j in range(n):
            x = rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x_list[j], Fvac_list[k]
            loglike[k,j] = loglik_obj_params(x, args) 


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure()
ax = fig.gca(projection='3d')
jet = plt.get_cmap('jet')

# Make data.
X, Y = np.meshgrid(Fvac_list, eps_x_list)
Z = -loglike - 600 # subtract approx. log likelihood constant 

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=jet, linewidth = 0)
ax.set_xlabel('Sunk Cost ratio ' r'$\kappa_x x / \kappa_V$')
ax.set_ylabel('Elasticity of Obsolescence 'r'$\epsilon_x$') 
ax.set_zlabel('Log Likelihood') 

#ax.set_zlim3d(0, Z.max())
ax.patch.set_facecolor('white')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
fig.set_size_inches(6, 5) 
fig.tight_layout()
#plt.savefig('plots/Labor_market/likelihood_plot.pdf')
 

plt.show()
                  

#%% Hessian

def return_loglike(x):
    rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac_factor = x
    #eps_x, Fvac = x

    Fvac_g = ss_est_copy['Fvac']
    vacK_g = ss_est_copy['vacK']
    
    sol = optimize.root(Vacancy_costs_calib, np.array([Fvac_g, vacK_g]),  method='hybr', args = (Fvac_factor,))
    (Fvac, vacK) = sol.x 
    if not sol.success:
        raise Exception("Solver did not succeed") 
                
    JV = Fvac * ss_est_copy['nPos']            
    Vac_costs = vacK  + Fvac * ss_est_copy['nPos'] **2
    JM  = (JV - (1-ss_est_copy['pMatch']) *(1-ss_est_copy['destrO']) * JV /(1+ss_est_copy['r']) + vacK) / ss_est_copy['pMatch']              

    ss_est_copy.update({'eps_x' : eps_x, 'Fvac' : Fvac, 'JV' : JV, 'JM' : JM, 'vacK' : vacK})
    G = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_est_copy, save=False, use_saved=True)

    dZ1 = rho_mup**(np.arange(Time))
    dY1, dU1, dV1 = G['Y'][var1] @ dZ1, G['U_rate'][var1] @ dZ1, G['V_rate'][var1] @ dZ1
    dX1 = np.stack([ dY1, dU1, dV1], axis=1)
    
    dZ2 = rho_r**(np.arange(Time))
    dY2, dU2, dV2 = G['Y'][var2] @ dZ2, G['U_rate'][var2] @ dZ2, G['V_rate'][var2] @ dZ2
    dX2 = np.stack([ dY2, dU2, dV2], axis=1)    
    
    dZ3 = rho_beta**(np.arange(Time))
    dY3, dU3, dV3 = G['Y'][var3] @ dZ3, G['U_rate'][var3] @ dZ3, G['V_rate'][var3] @ dZ3
    dX3 = np.stack([ dY3, dU3, dV3], axis=1)    
    
    
    dX = np.stack([dX1, dX2, dX3], axis=2)
    sigmas = np.array([sigma_mup, sigma_r, sigma_beta])
    Sigma = est.all_covariances(dX, sigmas) # burn-in for jit
    
    # calculate log=likelihood from this
    return est.log_likelihood(Y, Sigma)


test = return_loglike(estimates)


#%%
import numdifftools as nd

dfun = nd.Hessian(return_loglike, step = 1e-05)
#Hessian_mat = dfun([eps_x, Fvac])
Hessian_mat = dfun([rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta, eps_x, Fvac_factor])
print(Hessian_mat)



#%% Fisher 

obs = Y.flatten().size
I = - Hessian_mat
var = np.linalg.inv(I ) 
stds = np.sqrt( abs(np.diag(var)) ) 
var_covar = abs(np.linalg.inv(I ) )
# std. errors
print(stds)


# CI
CI_lvl = 1.96 
eps_x_CI = [eps_x - CI_lvl * stds[-2] , eps_x + CI_lvl * stds[-2]  ]
Fvac_CI = [Fvac_factor - CI_lvl * stds[-1] , Fvac_factor + CI_lvl * stds[-1]  ]

rho_mup_CI = [rho_mup - CI_lvl * stds[0] , rho_mup + CI_lvl * stds[0]  ]
rho_r_CI = [rho_r - CI_lvl * stds[1] , rho_r + CI_lvl * stds[1]  ]
rho_beta_CI = [rho_beta - CI_lvl * stds[2] , rho_beta + CI_lvl * stds[2]  ]

sigma_mup_CI = [sigma_mup - CI_lvl * stds[3] , sigma_mup + CI_lvl * stds[3]  ]
sigma_r_CI = [sigma_r - CI_lvl * stds[4] , sigma_r + CI_lvl * stds[4]  ]
sigma_beta_CI = [sigma_beta - CI_lvl * stds[5] , sigma_beta + CI_lvl * stds[5]  ]

print('CI')
print(rho_mup_CI, "\n", rho_r_CI)
print(rho_beta_CI, "\n", sigma_mup_CI)
print(sigma_r_CI, "\n", sigma_beta_CI)
print(eps_x_CI, "\n", Fvac_CI)


#%%
Time = 300 
IvarPE = 'Z'
IvarGE = 'mup'
GE_shock = 0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)


def comp_LM_models_GE(IvarGE, dZ_GE): 
    exogenous = [IvarGE]  
    unknowns = ['L', 'mc']
    targets = ['Asset_mkt', 'Labor_mkt']
    settings['Fvac_share'] = False 
    settings['Fvac_factor'] = Fvac_factor  
    settings['vac2w_costs'] = 0.05      
    settings['q_calib'] = False
    settings['destrO_share'] = 0.5    
    settings['theta_exp'] = 0
    settings['theta_adap'] = 0.7 

    use_sticky_exp = False  
    
    Asset_block = Asset_block_T
    #Asset_block = Asset_block_B
    if IvarGE == 'I_Z' : 
        prod_stuff = solved(block_list=[monetary1, pricing, ProdFunc, firm_investment2],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  
    else:
        prod_stuff = solved(block_list=[monetary1, pricing, ProdFunc, firm_investment2],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  
  

    # Full FR model with endo. sep delta^O
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
    
    
    ss_costly_FR_endo_destr = ss_calib('Solve', settings)
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    
    
    ss_costly_FR_endo_destr.update({'eps_x' : eps_x})
    ss_HH =   EGMhousehold.ss(**ss_costly_FR_endo_destr) 
    for key in ss_HH:
        ss_costly_FR_endo_destr[key] = ss_HH[key]
    
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endo_destr, save=False)
    
    dMat_costly_vac_FR_endo_destr = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    # Full FR model with exo sep.
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
  
    ss_costly_FR_exo_destr = ss_calib('Solve', settings)
    ss_costly_FR_exo_destr.update({'eps_x' : 0})
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
    ss_costly_FR_endo_destr_no_precaut.update({'eps_x' : eps_x})
    ss_HH =   EGMhousehold.ss(**ss_costly_FR_endo_destr_no_precaut) 
    for key in ss_HH:
        ss_costly_FR_endo_destr_no_precaut[key] = ss_HH[key]    
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend] 
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_costly_FR_endo_destr_no_precaut, save=False)    
    dMat_costly_vac_FR_endo_destr_no_precaut = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    # Full FR model with endo sep. no sunk costs 
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
    
    settings['Fvac_share'] = False 
    settings['Fvac_factor'] = 0
    
    ss_costly_FR_endo_destr_no_sunkcost = ss_calib('Solve', settings)
    ss_costly_FR_endo_destr_no_sunkcost.update({'eps_x' : 0})
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    ss_costly_FR_endo_destr_no_sunkcost.update({'eps_x' : 0})
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



def comp_LM_models_GE_figs_CI(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ssbase, dBase, std, betscale, plot_CI):
    pylab.rcParams.update(params)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    plot_hori = 60 
    
    Inits = np.empty([plot_hori])
    Inits[:] = 1
    if betscale:
        Inits[0] = 0.5
    else:
        Inits[0] = 1
        
    pl = 1.5
    
    lstyle_precaut = '--'
    lstyle_exosep = '-'
    
    fullmodel1 = 100 * dMat_costly_vac_FR_endo_destr['N'][:plot_hori]/ss_costly_FR_endo_destr['N']
    ax1.plot( fullmodel1, linewidth=pl, label = 'Full Model')
    ax1.plot(100 * dMat_costly_vac_FR_exo_destr['N'][:plot_hori]/ss_costly_FR_exo_destr['N'], linewidth=pl, label = 'Exo. Seperations', linestyle = lstyle_exosep)
    ax1.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['N'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['N'], linewidth=pl, label = 'No unemployment risk', color = 'Darkgreen', linestyle = lstyle_precaut)
    ax1.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['N'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['N'] * Inits, linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
        
    
    ax1.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax2.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax3.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    fullmodel2 = 100 * dMat_costly_vac_FR_endo_destr['V'][:plot_hori]/ss_costly_FR_endo_destr['V']
    ax2.plot( fullmodel2, linewidth=pl)
    ax2.plot(100 * dMat_costly_vac_FR_exo_destr['V'][:plot_hori]/ss_costly_FR_exo_destr['V'], linewidth=pl, linestyle = lstyle_exosep)
    ax2.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['V'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['V'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_precaut)
    ax2.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['V'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['V'] * Inits, linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    ax2.set_title('Vacancies')
    
    fullmodel3 = 100 * dMat_costly_vac_FR_endo_destr['q'][:plot_hori]/ss_costly_FR_endo_destr['q']
    ax3.plot( fullmodel3, linewidth=pl)
    ax3.plot(100 * dMat_costly_vac_FR_exo_destr['q'][:plot_hori]/ss_costly_FR_exo_destr['q'], linewidth=pl, linestyle = lstyle_exosep)
    ax3.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['q'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['q'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_precaut)
    ax3.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['q'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['q'] * Inits, linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    ax3.set_title('Job-finding rate')
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
    ax6.set_xlabel('quarters')
    ax6.set_ylabel('Pct. points Deviation from SS')
    ax6.set_title('Separation Rate')
    
    ax4.set_title('Consumption')
    ax5.set_title('Output')
    
    ax4.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax5.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    ax6.plot(np.zeros(plot_hori),  linestyle='--', linewidth=1, color='black')
    
    fullmodel4 = 100 * dMat_costly_vac_FR_endo_destr['CTD'][:plot_hori]/ss_costly_FR_endo_destr['C']
    ax4.plot( fullmodel4, linewidth=pl)
    ax4.plot(100 * dMat_costly_vac_FR_exo_destr['CTD'][:plot_hori]/ss_costly_FR_exo_destr['C'], linewidth=pl, linestyle = lstyle_exosep)
    ax4.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['CTD'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['C'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_precaut)
    ax4.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['CTD'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['C_agg'] , linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    fullmodel5 = 100 * dMat_costly_vac_FR_endo_destr['Y'][:plot_hori]/ss_costly_FR_endo_destr['Y']
    ax5.plot( fullmodel5, linewidth=pl)
    ax5.plot(100 * dMat_costly_vac_FR_exo_destr['Y'][:plot_hori]/ss_costly_FR_exo_destr['Y'], linewidth=pl, linestyle = lstyle_exosep)
    ax5.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['Y'][:plot_hori]/ss_costly_FR_endo_destr_no_precaut['Y'], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_precaut)
    ax5.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['Y'][:plot_hori]/ss_costly_FR_endo_destr_no_sunkcost['Y'] * Inits, linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    fullmodel6 = 100 * dMat_costly_vac_FR_endo_destr['destr'][:plot_hori]
    ax6.plot( fullmodel6, linewidth=pl)
    ax6.plot(np.zeros(plot_hori), linewidth=pl, linestyle = lstyle_exosep)
    ax6.plot(100 * dMat_costly_vac_FR_endo_destr_no_precaut['destr'][:plot_hori], color = 'Darkgreen', linewidth=pl, linestyle = lstyle_precaut)
    ax6.plot(100 * dMat_costly_vac_FR_endo_destr_no_sunkcost['destr'][:plot_hori], linewidth=pl, label = 'No Sunk Cost', color = 'firebrick', linestyle = 'dotted')
    
    plot_base = False
    
    if plot_base:
        markerstyle = 'dashdot'
        col_base = 'black'
        markersize_set = 1
        ax1.plot(100 * dBase['N'][:plot_hori]/ssbase['N'] * Inits, linewidth=pl, label = 'Basic HANK', linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax2.plot(100 * dBase['V'][:plot_hori]/ssbase['V'] * Inits, linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax3.plot(100 * dBase['q'][:plot_hori]/ssbase['q'] * Inits, color = col_base, linewidth=pl, linestyle = markerstyle, markersize = markersize_set)
        ax4.plot(100 * dBase['CTD'][:plot_hori]/ssbase['C'] , linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax5.plot(100 * dBase['Y'][:plot_hori]/ssbase['Y'] * Inits, linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)
        ax6.plot(np.zeros(plot_hori), linewidth=pl, linestyle = markerstyle, color = col_base, markersize = markersize_set)

    ax1.legend(loc='best',prop={'size': 6})
    ax1.set_title('Employment')

    x_axis = np.arange(plot_hori)

    CI_col = 'blue'
    #ci_val = 1.96
    ci_val = 1.645   
    #ci_val = 1.2   
    if plot_CI:
        ci1 = ci_val * std['N'][:plot_hori] / ss_costly_FR_endo_destr['N'] * 100
        ci2 = ci_val * std['V'][:plot_hori] / ss_costly_FR_endo_destr['V'] * 100
        ci3 = ci_val * std['q'][:plot_hori] / ss_costly_FR_endo_destr['q'] * 100
        ci4 = ci_val * std['C_agg'][:plot_hori] / ss_costly_FR_endo_destr['C_agg'] * 100
        ci5 = ci_val * std['Y'][:plot_hori] / ss_costly_FR_endo_destr['Y'] * 100
        ci6 = ci_val * std['destr'][:plot_hori]  * 100
        
        alpha_lvl = 0.2
        ax1.fill_between(x_axis, fullmodel1 + ci1, fullmodel1 - ci1, color=CI_col, alpha = alpha_lvl)
        ax2.fill_between(x_axis, fullmodel2 + ci2, fullmodel2 - ci2, color=CI_col, alpha = alpha_lvl)
        ax3.fill_between(x_axis, fullmodel3 + ci3, fullmodel3 - ci3, color=CI_col, alpha = alpha_lvl)
        ax4.fill_between(x_axis, fullmodel4 + ci4, fullmodel4 - ci4, color=CI_col, alpha = alpha_lvl)
        ax5.fill_between(x_axis, fullmodel5 + ci5, fullmodel5 - ci5, color=CI_col, alpha = alpha_lvl)
        ax6.fill_between(x_axis, fullmodel6 + ci6, fullmodel6 - ci6, color=CI_col, alpha = alpha_lvl)
    
    plt.gcf().set_size_inches(8, 5) 
    fig.tight_layout()
    
    return fig 




#%%
Time = 300 

def Delta_method_CI_IRFs(IvarGE, dZ_GE):
    
    settings['Fvac_share'] = False 
    settings['vac2w_costs'] = 0.05      
    settings['q_calib'] = False
    settings['destrO_share'] = 0.5    
    settings['SAM_model'] = 'Costly_vac_creation'
    settings['SAM_model_variant'] = 'FR'
    settings['endo_destrNO'] = False
    settings['endo_destrO'] = True
    
    settings['Fvac_factor'] = Fvac_factor
    
    exogenous = [IvarGE]
    

    
    prod_stuff = solved(block_list=[monetary1, pricing, ProdFunc, firm_investment2],
            unknowns=[ 'pi', 'Y'],
            targets=[  'nkpc' , 'ProdFunc_Res' ] )  
    
    ss_base = ss_calib('Solve', settings)
    dividend, LM = Choose_LM(settings)
    block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, dividend, Eq, destr_rate_lag] 
    
    ss_base.update({'eps_x' : eps_x })
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_base, save=True)
    dMat_base = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    eps_eps_x = 1e-04
    ss_base.update({'eps_x' : eps_x + eps_eps_x})
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_base, save=False)
    dMat_eps_x = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    eps_Fvac = 1e-04 
    settings['Fvac_factor'] = Fvac_factor + eps_Fvac
    ss_base = ss_calib('Solve', settings)
    ss_base.update({'eps_x' : eps_x })
    
    
    G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss_base, save=False)
    dMat_Fvac = FigUtils.Shock_mult(G_jac, dZ_GE, IvarGE)
    
    G_prime = {}
    G_prime['eps_x'] = {}
    G_prime['Fvac'] = {}
    
    for key in dMat and dMat_eps_x: 
        G_prime['eps_x'][key] = (dMat_eps_x[key] - dMat_base[key]) / eps_eps_x 
        G_prime['Fvac'][key] = (dMat_Fvac[key] - dMat_base[key])  / eps_Fvac 
        
    var_X = np.empty([2,2])
    var_X[0,0] = var_covar[-2,-2]
    var_X[1,1] = var_covar[-1,-1]
    var_X[0,1] = var_covar[-1,-2]  
    var_X[1,0] = var_X[0,1]

    
    IRF_std = {}
    
    for key in dMat and dMat_eps_x:
        G_prime_stack = np.stack((G_prime['eps_x'][key], G_prime['Fvac'][key]))
        IRF_std[key] = np.empty([Time])
    
        for j in prange(Time):
            IRF_std[key][j] =  np.sqrt( G_prime_stack[:,j] @ var_X @ G_prime_stack[:,j])  
        
    return IRF_std

IvarGE = 'mup'
GE_shock = 0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

IRF_std = Delta_method_CI_IRFs(IvarGE, dZ_GE)


#%%

IvarGE = 'mup'
GE_shock = 0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ss_base, dMat_base = comp_LM_models_GE(IvarGE, dZ_GE)
IRF_std = Delta_method_CI_IRFs(IvarGE, dZ_GE)
fig = comp_LM_models_GE_figs_CI(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost,ss_base, dMat_base, 0, False, False)
#plt.savefig('plots/Labor_market/general_amplification_markup_CI.pdf') 


#%%

IvarGE = 'beta_shock'
GE_shock = -0.01 * ss[IvarGE]
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ss_base, dMat_base = comp_LM_models_GE(IvarGE, dZ_GE)
IRF_std = Delta_method_CI_IRFs(IvarGE, dZ_GE)
fig = comp_LM_models_GE_figs_CI(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost,ss_base, dMat_base, IRF_std, True)



#%%
IvarGE = 'Z'
GE_shock = - 0.01 
dZ_PE, dZ_GE = LM_shocks(Time, IvarPE, IvarGE, ss, GE_shock)

ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost, ss_base, dMat_base = comp_LM_models_GE(IvarGE, dZ_GE)
IRF_std = Delta_method_CI_IRFs(IvarGE, dZ_GE)
fig = comp_LM_models_GE_figs_CI(ss_costly_FR_endo_destr, ss_costly_FR_exo_destr, ss_costly_FR_endo_destr_no_precaut, ss_costly_FR_endo_destr_no_sunkcost, dMat_costly_vac_FR_endo_destr, dMat_costly_vac_FR_exo_destr, dMat_costly_vac_FR_endo_destr_no_precaut, dMat_costly_vac_FR_endo_destr_no_sunkcost,ss_base, dMat_base, IRF_std, False)



