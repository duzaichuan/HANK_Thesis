
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from HANKSAM import *
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

#%% MPCs in steady state 

Time = 300  
dTuni = np.empty([Time])   
dTuni[:]  = 0
dTuni[0]  = 0.01 *ss['w'] 


ttt = np.arange(0,Time)
tvar = {'time' : ttt ,'Tuni' : ss['Tuni'] + dTuni} 
td =   EGMhousehold.td(ss, returnindividual = True, monotonic=True, **tvar)


plt.plot(dTuni, label='linear', linestyle='-', linewidth=2.5)
plt.show()
plt.plot(100 * ((td['C'][0:40]/ss['C'])-1), label='linear', linestyle='-', linewidth=2.5)
plt.show()


dINC = td['INC'][0]-ss['INC']
dCC =  (1+ss['VAT']) * (td['C'] - ss['C'])
 
agg_MPC = 1-(1-dCC/dINC)**4
temp = dCC/dINC
agg_MPC_yearly =  [sum(temp[i:i+4]) for i in range(0, len(temp), 2)]
print(1-(1-dCC[0]/dINC)**4, agg_MPC_yearly[0]  )
norw_data = [0.54, 0.16, 0.08, 0.05, 0.04, 0.03]
x_ = [0, 4, 8, 12, 16, 20]
#plt.plot(np.arange(0,21), agg_MPC[:21])
plt.plot(agg_MPC_yearly[:6], label = 'HANK Model')
plt.plot(norw_data, 'o', marker = "d", label = 'Empirical Estimate')
pylab.rcParams.update(params)   
plt.legend(loc="upper right")
plt.xlabel('Years')
plt.ylabel('Annual MPC')    
plt.legend(fontsize=10)
plt.gcf().set_size_inches(7/1.8, 6/1.8) 
plt.tight_layout()
#plt.savefig('plots/calibration/Mpc_compare_Norway.pdf')
plt.show()    

dINC_indi = dTuni[0]
dCC_indi =  (1+ss['VAT']) * (td['c'] - ss['c'])
MPC_indi = 1-(1-dCC_indi[0]/dINC_indi)**4 


#%%
  
hist = np.histogram(ss['a'].flatten(), bins=ss['nPoints'][2], range=None, normed=None, weights=ss['D'].flatten(), density=None)
  
fig, ax1 = plt.subplots( sharey =False)
  
c_norm = (1+ss['VAT']) * ss['c'] / ss['Inc'] 
m = (ss['Inc'] + ss['a_grid'][np.newaxis, :] * (1+ss['ra'])) / ss['Inc'] 


mean_beta = np.vdot(ss['pi_beta'],ss['beta'])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

nearest_ = find_nearest(ss['beta'], mean_beta)
if nearest_ == ss['nPoints'][0]:
    nearest_ = ss['nPoints'][0] -1
ne = ss['nPoints'][1]  
nA = ss['nPoints'][2] 
earnings_grid = np.repeat(ss['pi_ern'], ( nA))
earnings_grid = np.reshape(earnings_grid, (ne, nA)) 
c_norm_mean =  np.pad(np.sum( c_norm[0+ne*nearest_:ne*(1+nearest_),:] * earnings_grid , axis = 0 ), (1, 0),'constant')
c_norm_low  =  np.pad(np.sum( c_norm[0:ne,:] * earnings_grid , axis = 0 ) , (1, 0),'constant')
c_norm_high  =  np.pad(np.sum( c_norm[-1-ne:-1,:] * earnings_grid , axis = 0 ) , (1, 0),'constant')

m_norm_mean = np.pad(np.sum( m[0+ne*nearest_:ne*(1+nearest_),:] * earnings_grid , axis = 0 ), (1, 0),'constant')
m_norm_low = np.pad(np.sum( m[0:ne,:] * earnings_grid , axis = 0 ) , (1, 0),'constant')
m_norm_high = np.pad(np.sum( m[-1-ne:-1,:] * earnings_grid , axis = 0 ) , (1, 0),'constant')

m_norm = ss['a']
ax1.hist(m_norm.flatten(), bins=300,  weights = ss['D'].flatten(), color = 'grey', edgecolor  = 'black')
ax2 = ax1.twinx()
ax2.plot( ss['a_grid'], c_norm_mean[1:], label = 'Average patience')
ax2.plot( ss['a_grid'], c_norm_low[1:], label = 'Most Impatient')
ax2.plot( ss['a_grid'], c_norm_high[1:], label = 'Most Patient', color = 'darkgreen' )
ax1.set_xlim([min(m_norm.flatten()),10])
ax2.set_ylim([0,5])
ax2.legend(loc="best")

ax1.set_ylabel('Density', labelpad=10)
ax2.set_ylabel('Normalized Consumption', labelpad=10)
ax1.set_xlabel('Normalized Net Worth')
ax2.legend(fontsize=7)

pylab.rcParams.update(params)
ax2.grid(None)
plt.gcf().set_size_inches(7/1.8, 6/1.8) 
plt.tight_layout()
#plt.savefig('plots/calibration/Consumption_function.pdf')    
plt.show()         



#%% Interest rate shock 

dra = - 0.0025  * 0.6**(np.arange(Time))
tvar       = {'time' : ttt, 'ra' :ss['ra'] + dra }
td       =   EGMhousehold.td(ss,       returnindividual = True, monotonic=True, **tvar)  
#td       =   C_RA.td(ss,   **tvar)  


pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)
tplot = 21 
x_t = np.arange(0,tplot)


dcc = (td['C']/ss['C'] -1)*100
ax2.plot(x_t, dcc[:tplot])
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Change')    
ax2.set_title('Consumption Response')
ax2.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')

ax1.set_ylabel('Pct. Points Change')
ax1.plot(x_t, 100 * dra[:tplot], color = 'tab:blue')
ax1.set_title('Interest rate shock')
ax1.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')
ax1.set_xlabel('quarters')

M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)   

plt.gcf().set_size_inches(7,2.5) 
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
fig.tight_layout()  
#plt.savefig('plots/calibration/Consumption_interest_rate_response.pdf') 
plt.show()

#%% Decompose Consumption response into individual and distributional effect 

tplot = 21 
dC_part = np.empty([tplot])
dD_part = np.empty([tplot])
dC_part_I = np.empty([tplot])
dD_part_I = np.empty([tplot])

dTuni = np.empty([Time])   
dTuni[:]  = 0
dTuni[0]  = 0.01 *ss['w'] 

ttt = np.arange(0,Time)

tvar = {'time' : ttt ,'Tuni' : dTuni} 
td_tuni =   EGMhousehold.td(ss, returnindividual = True, monotonic=True, **tvar)

dcc_I = (td_tuni['C']/ss['C'] -1)*100

for j in range(tplot):
    dC_part[j] = np.vdot( (td['c'][j,:,:] - ss['c']),  ss['D']  )
    dD_part[j] = np.vdot( (td['D'][j,:,:] - ss['D']),  ss['c']  ) 
    dC_part_I[j] = np.vdot( (td_tuni['c'][j,:,:] - ss['c']),  ss['D']  )
    dD_part_I[j] = np.vdot( (td_tuni['D'][j,:,:] - ss['D']),  ss['c']  ) 
    
x_t = np.arange(0,tplot)
    
pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)

ax1.plot(x_t, 100 * dD_part_I / ss['C'], label = 'Distributional Effect')    
ax1.plot(x_t, 100 * dC_part_I / ss['C'], label = 'Individual Effect')  
ax1.plot(x_t, dcc_I[:tplot], label = 'Total', linestyle='--', color = 'darkgreen')  
ax1.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')  

ax2.plot(x_t, 100 * dD_part / ss['C'], label = 'Distributional Effect')    
ax2.plot(x_t, 100 * dC_part / ss['C'], label = 'Individual Effect')  
ax2.plot(x_t, dcc[:tplot], label = 'Total', linestyle='--', color = 'darkgreen')  
ax2.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')  


M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)    
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Change')

ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Change')
#ax2.legend(loc = 'best' )
ax1.set_title('Income shock')
ax2.set_title('Interest rate shock')

plt.gcf().set_size_inches(7,2.5)    
ax1.legend(loc = 'best' )
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
fig.tight_layout()  
#plt.savefig('plots/C_analysis/dC__decomp_dist.pdf')   
plt.show()

HANK_I_response = 100 * dC_part_I / td['C'][:tplot]
HANK_R_response = 100 * dC_part / td['C'][:tplot]
dC_dR_HANK = HANK_R_response[0]
HANK_MPC = HANK_I_response[0]


#%% HA repsonse vs TA/RA 

from scipy import optimize

def RA_tax(UINCAGG, w, ssAvgInc, cs_avg):    
    watax = avgTaxf(w , ssAvgInc, cs_avg) 
    UINCAGGatax = avgTaxf(UINCAGG, ssAvgInc, cs_avg)
    return watax, UINCAGGatax


@solved(unknowns=[ 'CR',  'A'], targets=['euler',  'budget_R'])
def C_TANK(ra, CR, L,  eis, A, Ttd, N,  Tss, VAT, INC, rstar, sHTM, Tuni, w, avg_beta, ssAvgInc, UINCAGG):       
    
    disc = (1/(1+rstar))
    
    #h = (w * e_grid_broad * (1-tax_broad**0.85)  / (vphi * (1+VAT))) ** frisch
    #INCAGG = N * w * h * (1-avgTaxf(h * w , ssAvgInc, cs_avg) )
    
    #disc = avg_beta 
    euler = CR ** (-1 / eis) -  disc * (1 + ra(+1)) * CR(+1) ** (-1 / eis)
    
    tN = 0.4 * w  
    tU = 0.3 * INC
    INCAGG = N * w * (1-tN) + (1-N) * UINCAGG * (1-tU)
    budget_R = CR * (1+VAT) + A - ((1 + ra) * A(-1) + (INCAGG + Tuni + Ttd + Tss)  )
    
    C_HtM = (INCAGG + Ttd + Tuni + Tss)  / (1+VAT)
    C = C_HtM * sHTM + CR * (1-sHTM)
    return euler, budget_R, C, C_HtM  


@solved(unknowns=['CR_N', 'A_N', 'CR_U', 'A_U'], targets=['eulerN', 'eulerU', 'budget_RN','budget_RU'])
def C_TANK_U(ra, CR_N, CR_U, L,  eis, A_N, A_U, Ttd, N,  Tss, VAT, INC, rstar, sHTM, Tuni, w, avg_beta, ssAvgInc, UINCAGG, destr, q, disc):       
    
    #disc = (1/(1+rstar))
    
    #h = (w * e_grid_broad * (1-tax_broad**0.85)  / (vphi * (1+VAT))) ** frisch
    #INCAGG = N * w * h * (1-avgTaxf(h * w , ssAvgInc, cs_avg) )
    
    #disc = avg_beta 
    eulerN = CR_N ** (-1 / eis) -  disc * (1 + ra(+1)) * ((1 - destr * (1-q)) *CR_N(+1) ** (-1 / eis) + destr * (1-q) * CR_U(+1) ** (-1 / eis))
    eulerU = CR_U ** (-1 / eis) -  disc * (1 + ra(+1)) * (q *CR_N(+1) ** (-1 / eis) + (1-q) * CR_U(+1) ** (-1 / eis))
    

    tN = 0.4   
    tU = 0.3 
    INCAGG = N * w * (1-tN) + (1-N) * UINCAGG * (1-tU)
    
    budget_RN = CR_N * (1+VAT) + A_N - ((1 + ra) * A_N(-1) + (w * (1-0.4) + Tuni + Ttd + Tss)  )
    budget_RU = CR_U * (1+VAT) + A_U - ((1 + ra) * A_U(-1) + (UINCAGG * (1-0.4)/0.05 + Tuni + Ttd + Tss)  )
    
    C_HtM = (INCAGG + Ttd + Tuni + Tss)  / (1+VAT)
    CR = N * CR_N + (1-N) * CR_U
    C = C_HtM * sHTM + CR * (1-sHTM)

    return eulerN, eulerU, budget_RN, budget_RU, C, C_HtM  

@simple
def C_TANK_U_nonlin(ra, CR_N, CR_U, L,  eis, A_N, A_U, Ttd, N,  Tss, VAT, INC, rstar, sHTM, Tuni, w, avg_beta, ssAvgInc, UINCAGG, destr, q, disc):           
    #h = (w * e_grid_broad * (1-tax_broad**0.85)  / (vphi * (1+VAT))) ** frisch
    #INCAGG = N * w * h * (1-avgTaxf(h * w , ssAvgInc, cs_avg) )

    eulerN = CR_N ** (-1 / eis) -  disc * (1 + ra(+1)) * ((1 - destr * (1-q)) *CR_N(+1) ** (-1 / eis) + destr * (1-q) * CR_U(+1) ** (-1 / eis))
    eulerU = CR_U ** (-1 / eis) -  disc * (1 + ra(+1)) * (q *CR_N(+1) ** (-1 / eis) + (1-q) * CR_U(+1) ** (-1 / eis))
    
    tN = 0.4   
    tU = 0.3 
    INCAGG = N * w * (1-tN) + (1-N) * UINCAGG * (1-tU)
    
    budget_RN = CR_N * (1+VAT) + A_N - ((1 + ra) * A_N(-1) + (w * (1-0.4) + Tuni + Ttd + Tss)  )
    budget_RU = CR_U * (1+VAT) + A_U - ((1 + ra) * A_U(-1) + (UINCAGG * (1-0.4) + Tuni + Ttd + Tss)  )
    
    C_HtM = (INCAGG + Ttd + Tuni + Tss)  / (1+VAT)
    CR = N * CR_N + (1-N) * CR_U
    C = C_HtM * sHTM + CR * (1-sHTM)

    return eulerN, eulerU, budget_RN, budget_RU, C, C_HtM  

@solved(unknowns=[ 'CR'], targets=['euler'])
def C_TANK_R(ra, CR, L,  eis, A, Ttd, N,  Tss, VAT, INC, rstar, sHTM, Tuni, w, avg_beta, UINCAGG):           
    disc = (1/(1+rstar))
    euler = CR ** (-1 / eis) -  disc * (1 + ra(+1)) * CR(+1) ** (-1 / eis)
    
    tN = 0.4 * w  
    tU = 0.3 * UINCAGG
    INCAGG = N * w * (1-tN) + (1-N) * UINCAGG * (1-tU)
    
    budget_R = CR * (1+VAT) + A - ((1 + ra) * A(-1) + (INCAGG + Tuni + Ttd + Tss)  )
    
    C_HtM = (INCAGG + Ttd + Tuni + Tss)  / (1+VAT)
    C = C_HtM * sHTM + CR * (1-sHTM)
    return euler, C, C_HtM  


def MPC_calib(x, *args):
    sHtM = x
    ss_ = args[0] 
    dTuni = args[1] 
    HANK_MPC = args[2] 

    
    ss_['sHTM'] = sHtM
    ss_['CR'] = ss_['C'] 
    
    td_TANK =   C_TANK.td(ss_, Tuni = dTuni)
 
    #dCC =  (1+ss_['VAT']) * (td_TANK['C'] - td_TANK['C'][200])

    dCC = 100 * (td_TANK['C']  / td_TANK['C'][Time-50]-1)
    
    
    return dCC[0] - HANK_MPC

def R_calib(x, *args):
    ra = x
    ss_1 = args[0] 
    ddd = args[1] 
    dC_dR_HANK = args[2] 
    ss_1.update({ 'ra' : ra, 'rstar' : ra})     
    ssCR   =  (ra * ss_1['A'] + (ss_1['INC'] + ss_1['Tuni'] + ss_1['Ttd']  + ss_1['Tss'] ))/  (1+ss_1['VAT']) 
    ss_1.update({'CR' :  ssCR})    
    td_TANK_ra =   C_TANK_R.td(ss_1, ra = ra + ddd)

    dC_dR_TANK = (td_TANK_ra['C']/ td_TANK_ra['C'][Time-50] -1)*100
    return dC_dR_HANK - dC_dR_TANK[0]


sHTM_g = 0.2
ss['CR'] = ss['C'] * (1-sHTM_g) 
ss['sHTM'] = sHTM_g
ss['avg_beta'] = np.vdot(ss['beta'], ss['pi_beta'])
re_calib_beta = False

ss_copy =  ss.copy()  

args = (ss_copy, dTuni, HANK_MPC)
res_HtM = optimize.toms748(MPC_calib, 0,1, args = args)


ss['sHTM'] = res_HtM
ss_copy['sHTM'] = res_HtM
td_TANK =   C_TANK.td(ss_copy, Tuni = dTuni)
dCC_TANK =  (td_TANK['C']/ td_TANK['C'][Time-50] -1)*100

ss_copy_ra =  ss_copy.copy() 
ddd = - 0.0025  * 0.6**(np.arange(Time))


args = (ss_copy_ra, ddd, dC_dR_HANK)
res_ra = optimize.toms748(R_calib, 0.005,3, args = args)

td_ra       = C_TANK.td(ss_copy, ra = ss['ra'] + ddd)
dCC_TANK_ra = (td_ra['C']/ td_ra['C'][Time-50] -1)*100

 
ss_copy_ra.update({'rstar' : res_ra, 'ra' :  res_ra})

td_ra_calibrated       = C_TANK_R.td(ss_copy_ra, ra = ss_copy_ra['ra'] + ddd)
dCC_TANK_ra_calibrated = (td_ra_calibrated['C']/ td_ra_calibrated['C'][Time-50] -1)*100

pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)

ax1.plot(x_t, HANK_I_response, label = 'HA - Individual Effect', color = '#348ABD')  
ax1.plot(x_t, dCC_TANK[:tplotl], label = 'TA', linestyle='--', color = 'darkgreen')  
ax1.legend(loc = 'best' )
ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')

ax2.plot(x_t, HANK_R_response, label = 'HA - Individual Effect', color = '#348ABD')  
ax2.plot(x_t, dCC_TANK_ra[:tplotl], label = 'TA - HA calibration', linestyle='dashdot', color = 'firebrick')  
ax2.plot(x_t, dCC_TANK_ra_calibrated[:tplotl], label = 'TA - re-calibrated', linestyle='--', color = 'darkgreen')  
ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')
ax2.legend(loc = 'best' )

M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)    
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Change')

ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Change')
ax2.legend(loc = 'best' )
ax1.set_title('Income shock')
ax2.set_title('Interest rate shock')


plt.gcf().set_size_inches(7,2.5) 
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
fig.tight_layout()  
#plt.savefig('plots/C_analysis/TANK_vs_HANK_behavoiral_decomp.pdf')   
plt.show()


#%% Decomposition of employment and job-finding rate shock 

ss_U =  ss_copy.copy() 
ss_U.update({'CR_N' : ss['CN'], 'CR_U' :  ss['CS'], 'A_N' :  ss['AN'], 'A_U' :  ss['AS'], 'disc' : 1/(1+ss['rstar'])})
dN =  0.01 *ss['N'] * 0.8 ** (np.arange(Time))
tvar = {'time' : ttt ,'N' : ss['N'] + dN} 
td_N =   EGMhousehold.td(ss, returnindividual = True, monotonic=True, **tvar)
Cvar = 'c'
dN_ = ss['N'] + shift(dN, 0, cval=0)
Dtd = N_mult('D', ss, td_N, dN_)
dC_part_N, dD_part_N = FigUtils.HA_Decomp(ss, td_N, tplotl, Cvar, Dtd)


td_N_calibrated       = C_TANK_R.td(ss_copy_ra, N = ss['N'] + dN)
dCC_TANK_N_calibrated = (td_N_calibrated['C'][:tplotl]/ td_N_calibrated['C'][200] -1)*100
dCC_HANK_N = (dC_part_N / ss['C'] )*100


dq =  0.01  * 0.8 ** (np.arange(Time))
tvar = {'time' : ttt ,'Eq' : ss['q'] + dq} 
td_q =   EGMhousehold.td(ss, returnindividual = True, monotonic=True, **tvar)
Cvar = 'ctd'
dC_part_q, dD_part_q = FigUtils.HA_Decomp(ss, td_q, tplotl, Cvar, td_q['D'])

        
dcc_N = (dD_part_N + dC_part_N)/ss['C'] *100
dcc_q = (td_q['C']/ss['C'] -1)*100

pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)

ax1.plot(x_t, 100 * dD_part_N / ss['C'], label = 'Distributional Effect')    
ax1.plot(x_t, 100 * dC_part_N / ss['C'], label = 'Individual Effect')  
ax1.plot(x_t, dcc_N[:tplotl], label = 'Total', linestyle='--', color = 'darkgreen')  
ax1.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')  

ax2.plot(x_t, 100 * dD_part_q / ss['C'], label = 'Distributional Effect')    
ax2.plot(x_t, 100 * dC_part_q / ss['C'], label = 'Individual Effect')  
ax2.plot(x_t, dcc_q[:tplotl], label = 'Total', linestyle='--', color = 'darkgreen')  
ax2.plot(np.zeros(tplotl),  linestyle='--', linewidth=1, color='black')  


M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)    
ax2.set_xlabel('quarters')
ax2.set_ylabel('Pct. Change')

ax1.set_xlabel('quarters')
ax1.set_ylabel('Pct. Change')
ax2.legend(loc = 'best' )
ax1.set_title('Employment shock')
ax2.set_title('Finding rate shock')

plt.gcf().set_size_inches(7,2.5)    
#ax1.legend(loc = 'center' )
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})    
fig.tight_layout()  
#plt.savefig('plots/C_analysis/dC__decomp_dist_N_q.pdf')   
plt.show()




#%% Expected vs Unexpected shocks for interest rate and job-finding rate 

Time = 200
t = np.arange(0,Time)

dra1 = - 0.0025  * 0.6**(np.arange(Time))
raTot = sum(dra1)
dra2 = raTot  * 0.8**(np.arange(Time)) / sum(0.8**(np.arange(Time)))
dra3 = raTot  * 0.9**(np.arange(Time)) / sum(0.9**(np.arange(Time)))

dq1 = 0.05 * ss['q'] * 0.6**(np.arange(Time))
qTot = sum(dq1)
dq2 = qTot  * 0.8**(np.arange(Time)) / sum(0.8**(np.arange(Time)))
dq3 = qTot  * 0.9**(np.arange(Time)) / sum(0.9**(np.arange(Time)))

ra_list = [dra1, dra2, dra3]
q_list = [dq1, dq2, dq3]    
dC_ra = []
dC_q = []
for j in range(3):
    tvar       = {'time' : t, 'ra' :ss['ra'] + ra_list[j] }
    td       =   EGMhousehold.td(ss,       returnindividual = False, monotonic=True, **tvar)          
    dC_ra.append((td['C']/ss['C'] -1)*100) 
    tvar       = {'time' : t, 'Eq' : ss['q'] + q_list[j] }
    td       =   EGMhousehold.td(ss,       returnindividual = False, monotonic=True, **tvar)          
    dC_q.append((td['C']/ss['C'] -1)*100) 


pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)
tplot = 21 
x_t = np.arange(0,tplot)

ax1.set_ylabel('Pct. Points Change')  
ax1.plot(x_t, 100*dra1[:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.6)))
ax1.plot(x_t, 100*dra2[:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.8)))
ax1.plot(x_t, 100*dra3[:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.9)), color = 'darkgreen')
ax1.set_title('Interest rate shock')
ax1.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')
ax1.set_xlabel('quarters')
ax1.legend(loc = 'best'  )

ax2.set_ylabel('Pct. Change')  
ax2.plot(x_t, 100*dq1[:tplot]/ss['q'], label = r'$\rho$ = %s' %float("{:.1f}".format(0.6)))
ax2.plot(x_t, 100*dq2[:tplot]/ss['q'], label = r'$\rho$ = %s' %float("{:.1f}".format(0.8)))
ax2.plot(x_t, 100*dq3[:tplot]/ss['q'], label = r'$\rho$ = %s' %float("{:.1f}".format(0.9)), color = 'darkgreen')
ax2.set_title('Job-finding rate shock')
ax2.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')
ax2.set_xlabel('quarters')

M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)   
    
plt.gcf().set_size_inches(7,2.5) 
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
fig.tight_layout()  
#plt.savefig('plots/calibration/ra_q_shocks.pdf') 
plt.show()    

pylab.rcParams.update(params)
fig, ((ax1, ax2)) = plt.subplots(1,2)
tplot = 21 
x_t = np.arange(0,tplot)

ax1.set_ylabel('Pct. Change in C')  
ax1.plot(x_t, dC_ra[0][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.6)))
ax1.plot(x_t, dC_ra[1][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.8)))
ax1.plot(x_t, dC_ra[2][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.9)), color = 'darkgreen')
ax1.set_title('Interest rate shock')
ax1.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')
ax1.set_xlabel('quarters')
ax1.legend(loc = 'best'  )
yticks = ticker.MaxNLocator(5)
ax1.yaxis.set_major_locator(yticks)

ax2.set_ylabel('Pct. Change in C')  
ax2.plot(x_t, dC_q[0][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.6)))
ax2.plot(x_t, dC_q[1][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.8)))
ax2.plot(x_t, dC_q[2][:tplot], label = r'$\rho$ = %s' %float("{:.1f}".format(0.9)), color = 'darkgreen')
ax2.set_title('Job-finding rate shock')
ax2.plot(np.zeros(tplot),  linestyle='--', linewidth=1, color='black')
ax2.set_xlabel('quarters')

yticks = ticker.MaxNLocator(5)
ax2.yaxis.set_major_locator(yticks)   

M = 5
xticks = ticker.MaxNLocator(M)
ax2.xaxis.set_major_locator(xticks)
ax1.xaxis.set_major_locator(xticks)   

plt.gcf().set_size_inches(7,2.5) 
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
fig.tight_layout()  
#plt.savefig('plots/calibration/ra_q_shocks_C.pdf') 
plt.show()


#%% steady state distributional statistics 

nPoints = ss['nPoints']
beta = ss['beta']

D      = ss['D']
A_dist = ss['a']
MPC    = MPC_indi

# Wealth Distributional statistics
weight = D.flatten()
wealthdata = A_dist.flatten()  

p01 = weighted_quantile(wealthdata, 0.01,  sample_weight=weight)
p10 = weighted_quantile(wealthdata, 0.1,  sample_weight=weight)
p30 = weighted_quantile(wealthdata, 0.3,  sample_weight=weight)
p50 = weighted_quantile(wealthdata, 0.5,  sample_weight=weight)    
p70 = weighted_quantile(wealthdata, 0.7,  sample_weight=weight)    
p90 = weighted_quantile(wealthdata, 0.9,  sample_weight=weight)       
p99 = weighted_quantile(wealthdata, 0.99,  sample_weight=weight)

p01_index = np.nonzero(A_dist <= p01)   
p10_index = np.nonzero(A_dist<=p10)
p30_index = np.nonzero(np.logical_and(A_dist>p10, A_dist<=p30))
p50_index = np.nonzero(np.logical_and(A_dist>p30, A_dist<=p50)) 
p70_index = np.nonzero(np.logical_and(A_dist>p50, A_dist<=p70))
p90_index = np.nonzero(np.logical_and(A_dist>p70, A_dist<=p90))    
top10_index = np.nonzero(A_dist>=p90)   
top1_index = np.nonzero(A_dist>=p99)   

MPC01  =  np.average(MPC[p01_index], weights = D[p01_index])
MPC10  =  np.average(MPC[p10_index], weights = D[p10_index])
MPC30  =  np.average(MPC[p30_index], weights = D[p30_index])
MPC50  =  np.average(MPC[p50_index], weights = D[p50_index])
MPC70  =  np.average(MPC[p70_index], weights = D[p70_index])
MPC90  =  np.average(MPC[p90_index], weights = D[p90_index])
MPCt90  =  np.average(MPC[top10_index], weights = D[top10_index])
MPCt99  =  np.average(MPC[top1_index], weights = D[top1_index])
print('MPCs', MPC01, MPC10, MPC30, MPC50, MPC70, MPC90, MPCt90, MPCt99 )


beta_full = np.reshape(np.broadcast_to(beta[:, np.newaxis, np.newaxis, np.newaxis], (nPoints[0],nPoints[1], 2, nPoints[2])), (nPoints[1]*nPoints[0]*2, nPoints[2]))    
disc01  = np.average(beta_full[p01_index], weights = D[p01_index])
disc10 = np.average(beta_full[p10_index], weights = D[p10_index])
disc30 = np.average(beta_full[p30_index], weights = D[p30_index])
disc50 = np.average(beta_full[p50_index], weights = D[p50_index])
disc70 = np.average(beta_full[p70_index], weights = D[p70_index])
disc90 = np.average(beta_full[p90_index], weights = D[p90_index])
disct90 = np.average(beta_full[top10_index], weights = D[top10_index])
disct99 = np.average(beta_full[top1_index], weights = D[top1_index])

print('Dics', disc01, disc10, disc30, disc50, disc70, disc90, disct90, disct99 )    

N_full = np.reshape( np.stack((np.zeros((nPoints[1]*nPoints[0], nPoints[2])), np.ones((nPoints[1]*nPoints[0], nPoints[2]))), axis=-1), (nPoints[1]*nPoints[0]*2, nPoints[2]))  
N_Correct = np.reshape( np.stack((ss['N']+np.zeros((nPoints[1]*nPoints[0], nPoints[2])), (1-ss['N'])*np.ones((nPoints[1]*nPoints[0], nPoints[2]))), axis=-1), (nPoints[1]*nPoints[0]*2, nPoints[2]))  
D_ = D * N_Correct
N01  = np.average(N_full[p01_index], weights = D_[p01_index] )
N10 = np.average(N_full[p10_index], weights = D_[p10_index])
N30 = np.average(N_full[p30_index], weights = D_[p30_index])
N50 = np.average(N_full[p50_index], weights = D_[p50_index])    
N70 = np.average(N_full[p70_index], weights = D_[p70_index])
N80 = np.average(N_full[p90_index], weights = D_[p90_index])   
Nt90 = np.average(N_full[top10_index], weights = D_[top10_index])
Nt99 = np.average(N_full[top1_index], weights = D_[top1_index]) 
  
print('Unemployment rate', N01, N10, N30, N50, N70, N80, Nt90,  Nt99)  

e_full = np.reshape(np.broadcast_to(ss['e_grid'][np.newaxis, :, np.newaxis, np.newaxis], (nPoints[0],nPoints[1], 2, nPoints[2])), (nPoints[1]*nPoints[0]*2, nPoints[2]))    
e01  = np.average(e_full[p01_index], weights = D[p01_index])
e10 = np.average(e_full[p10_index], weights = D[p10_index])
e30 = np.average(e_full[p30_index], weights = D[p30_index])
e50 = np.average(beta_full[p50_index], weights = D[p50_index])
e70 = np.average(e_full[p70_index], weights = D[p70_index])
e90 = np.average(e_full[p90_index], weights = D[p90_index])
et90 = np.average(e_full[top10_index], weights = D[top10_index])
et99 = np.average(e_full[top1_index], weights = D[top1_index])

e_dist = np.stack((ss['N'] * ss['pi_ern'], (1-ss['N']) * ss['pi_ern']), axis=-1)
e_grid1 = np.stack((ss['w'] * ss['e_grid'], ss['UINCAGG'] * ss['e_grid']/(1-ss['N'])), axis=-1)
e01sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e01)
e10sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e10)
e30sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e30)
e50sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e50)
e70sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e70)
e90sc  = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), e90)
et90sc = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), et90)
et99sc = weighted_percentile_of_score(e_grid1.flatten(), e_dist.flatten(), et99)

print('earnings skill', e01sc, e10sc, e30sc, e50sc, e70sc, e90sc, et90sc, et99sc )    

# Wealth gini 
print('Wealth gini',  gini(A_dist.flatten(), w=D.flatten()))

# Income Distributional statistics   
Idata = ss['Inc'].flatten() + ss['taxN'].flatten() + ss['taxS'].flatten()
Idist = ss['Inc'] + ss['taxN'] + ss['taxS']

p01 = weighted_quantile(Idata, 0.01,  sample_weight=weight)
p10 = weighted_quantile(Idata, 0.1,  sample_weight=weight)
p30 = weighted_quantile(Idata, 0.3,  sample_weight=weight)
p50 = weighted_quantile(Idata, 0.5,  sample_weight=weight)    
p70 = weighted_quantile(Idata, 0.7,  sample_weight=weight)    
p90 = weighted_quantile(Idata, 0.9,  sample_weight=weight)       
p99 = weighted_quantile(Idata, 0.99,  sample_weight=weight)

p01_index = np.nonzero(Idist <= p01)   
p10_index = np.nonzero(Idist<=p10)
p30_index = np.nonzero(np.logical_and(Idist>p10, Idist<=p30))
p50_index = np.nonzero(np.logical_and(Idist>p30-0.001, Idist<=p50+0.001)) 
p70_index = np.nonzero(np.logical_and(Idist>p50, Idist<=p70))
p90_index = np.nonzero(np.logical_and(Idist>p70, Idist<=p90))    
top10_index = np.nonzero(Idist>=p90)   
top1_index = np.nonzero(Idist>=p99)  

MPC01  =  np.average(MPC[p01_index], weights = D[p01_index])
MPC10  =  np.average(MPC[p10_index], weights = D[p10_index])
MPC30  =  np.average(MPC[p30_index], weights = D[p30_index])
MPC50  =  np.average(MPC[p50_index], weights = D[p50_index])
MPC70  =  np.average(MPC[p70_index], weights = D[p70_index])
MPC90  =  np.average(MPC[p90_index], weights = D[p90_index])
MPCt90  =  np.average(MPC[top10_index], weights = D[top10_index])
MPCt99  =  np.average(MPC[top1_index], weights = D[top1_index])
print('MPCs', MPC01, MPC10, MPC30, MPC50, MPC70, MPC90, MPCt90, MPCt99 )
 


Taxable_inc = ss['Inc'] + ss['taxN'] + ss['taxS']

taxrates = avgTaxf(Taxable_inc, ss['ssAvgInc'], ss['cs_avg']) 
tax01   = np.average(taxrates[p01_index], weights = D[p01_index])
tax10  = np.average(taxrates[p10_index], weights = D[p10_index])
tax30 =  np.average(taxrates[p30_index], weights = D[p30_index])
tax50 =  np.average(taxrates[p50_index], weights = D[p50_index])         
tax70  = np.average(taxrates[p70_index], weights = D[p70_index])
tax90 =  np.average(taxrates[p90_index], weights = D[p90_index])
taxt90 =  np.average(taxrates[top10_index], weights = D[top10_index])     
taxt99 =  np.average(taxrates[top1_index], weights = D[top1_index]) 

print('Tax rates', tax01, tax10, tax30, tax50, tax70, tax90, taxt90, taxt99 )       

N_full = np.reshape( np.stack((np.zeros((nPoints[1]*nPoints[0], nPoints[2])), np.ones((nPoints[1]*nPoints[0], nPoints[2]))), axis=-1), (nPoints[1]*nPoints[0]*2, nPoints[2]))  
N01  = np.average(N_full[p01_index], weights = D_[p01_index])
N10 = np.average(N_full[p10_index], weights = D_[p10_index])
N30 = np.average(N_full[p30_index], weights = D_[p30_index])
N50 = np.average(N_full[p50_index], weights = D_[p50_index])    
N70 = np.average(N_full[p70_index], weights = D_[p70_index])
N80 = np.average(N_full[p90_index], weights = D_[p90_index])   
Nt90 = np.average(N_full[top10_index], weights = D_[top10_index])
Nt99 = np.average(N_full[top1_index], weights = D_[top1_index]) 
  
print('Unemployment rate', N01, N10, N30, N50, N70, N80, Nt90,  Nt99)  
  
# Income gini 
I_pretax = Taxable_inc
print('Incine gini (pre-tax)',  gini(I_pretax.flatten(), w=D.flatten()))
I_posttax = ss['Inc']
print('Incine gini (post-tax)',  gini(I_posttax.flatten(), w=D.flatten()))



#%% General Equbirium shocks and analysis 


dividend, LM = Choose_LM(settings)

Asset_block_G = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, dividend, EGMhousehold,  arbitrage, MutFund, Fiscal_stab_G,  aggregate],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

Asset_block_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res,  arbitrage, EGMhousehold, MutFund, Fiscal_stab_T, aggregate],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

Asset_block_B = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, dividend, EGMhousehold,  arbitrage, MutFund,  aggregate],
                unknowns=[ 'B',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

Asset_block_only_T = solved(block_list=[ fiscal_rev, fiscal_exp, B_res, EGMhousehold,  arbitrage, MutFund,  aggregate, HHTransfer],
                unknowns=[ 'uT',  'ra'],
                targets=[  'B_res','MF_Div_res'] )  

Asset_block_no_G = solved(block_list=[dividend, EGMhousehold,  arbitrage, MutFund,  aggregate],
                unknowns=[   'ra'],
                targets=[  'MF_Div_res'] )  

prod_stuff = solved(block_list=[monetary, pricing, ProdFunc, firm_investment],
                unknowns=[ 'pi', 'Y'],
                targets=[  'nkpc' , 'ProdFunc_Res' ] )  

Time = 300   

# markup shock 
Ivar = 'mup'
exogenous = [Ivar]  
rhos = 0.6
dZ =  0.1 *ss[Ivar] * rhos**(np.arange(Time))


exogenous = [Ivar]   
unknowns = ['L', 'mc']
targets = ['Asset_mkt', 'Labor_mkt']
block_list = [LM, Asset_block_T, prod_stuff, Asset_mkt_clearing, Labor_mkt_clearing, destr_rate_lag, dividend, Eq] 
 
G_jac = jac.get_G(block_list, exogenous, unknowns, targets,  Time, ss, save=False)
dMat = FigUtils.Shock_mult(G_jac, dZ, Ivar) # multiply Jacobian and shock 

tplot = 30 
fig = FigUtils.FigPlot3x3_new(ss, dMat, Ivar, tplot, dZ)
plt.show() 

# Decompose Consumption
fig = FigUtils.C_decomp(ss, dMat, 60, Time, EGMhousehold, False, 'CTD')
#plt.rcParams.update({'axes.titlesize': 'x-large'})
#plt.rcParams.update({'axes.labelsize': 'small'})
#plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
#plt.savefig('plots/calibration/C_decomp_main_T.pdf') 



#%% lifetime consumption equiv. weflare 
  
ft = Time 
ttt = np.arange(0,Time)

td, str_lst, tvar_orgshock = FigUtils.return_indi_HH(ss, dMat, Time, EGMhousehold)

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
    

# decomposed dC by employment status 
c_peak_decomped = FigUtils.dC_decomp_p0(css, ss, dec, A_dist, Atd, str_lst, deciles, tvar_orgshock, dMat, EGMhousehold)
dc_decomped_N, dc_decomped_U = FigUtils.C_decomp_peak_change_by_N(css, ss, dec, A_dist, Atd, str_lst, deciles, peakdC_index, tvar_orgshock, dMat, EGMhousehold)

# Decomposed dC0
fig = FigUtils.dC_decomp_and_C_deciles(c_peak_decomped, ss)
#plt.savefig('plots/C_analysis/peak_c_by_decile.pdf')    

# Decomposed dC0 by employment 
fig = FigUtils.dC_decomp_by_N(dc_decomped_N, dc_decomped_U, ss)
#plt.savefig('plots/C_analysis/dc_by_decile_decomp_by_N.pdf')   

# Welfare by Employment 
cons_equiv_N, cons_equiv_U = FigUtils.Welfare_equiv_by_N(ss, td)
fig = FigUtils.Welfare_equiv_by_N_Fig(ss, cons_equiv_N, cons_equiv_U, True)
#plt.savefig('plots/C_analysis/C_equiv_by_N.pdf') 





