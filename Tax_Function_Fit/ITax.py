# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:47:13 2020

@author: Nicolai
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Budget data 
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, date2num
import numpy as np
import statsmodels.api as sm
import scipy.interpolate 
import pickle
from scipy.optimize import curve_fit


#os.chdir('C:\\Users\\Nicolai\\Dropbox\\Thesis_Ideas\\MPC\\tax_function')

import matplotlib.pylab as pylab

plt.style.use('ggplot')
%config InlineBackend.figure_format = 'retina'
params = {'legend.fontsize': 'small',
          'figure.figsize': (7/1.8, 5/1.8),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}

#%% Data

# https://www.skm.dk/english/facts-and-figures/average-tax-rates-for-all-taxpayers

# average wage 2013  39.000 om måneden  
df = pd.read_excel (r'taxes.xlsx')

w = df[["lon"]].to_numpy()   
w = w / (12 * 39)
avgT  = df["gns. Skattesats"].to_numpy()   
margT =  df["marginal skattesats"].to_numpy()   


deg = 4
#z_avg  = np.polyfit(w[:,0], avgT/100, deg)

def f_func(x, a, b, c, d, e, g):
    cutoff = 2 
    iconst = np.greater(x, cutoff).astype(int)
    
    y = (a + b * x + c *x**2 + d *x**3 + e *x**4) * (1-iconst)  + g * iconst
    return y 

popt, pcov = curve_fit(f_func, w[:,0], avgT/100) 


z_marg = np.polyfit(w[:,0], margT/100, deg)



#np.save('avgtax_fit.npy', z_avg)
#np.save('margtax_fit.npy', z_marg)


#%% Plot
xa = np.linspace(0,3,1000)
xa1 = np.linspace(0,2,1000)


pylab.rcParams.update(params)   
plt.plot(100 * xa, f_func(xa, *popt), label='Fitted')
plt.plot(100 * xa1,avgT[:-1]/100, label='Data') 
plt.legend()
plt.xlabel('Pct. of Average Income')
plt.ylabel('Average Tax rate')
plt.tight_layout()
plt.savefig('tax_function1.pdf')    
plt.show()   
  

def cs_avg(x):
    cutoff = 2 
    iconst = np.greater(x, cutoff).astype(int)
    
    y = (popt[0] + popt[1] * x + popt[2] *x**2 + popt[3] *x**3 + popt[4] *x**4) * (1-iconst)  + popt[5] * iconst
    return y 

   
with open('cs_avg.pickle', 'wb') as f:
    pickle.dump(cs_avg, f, pickle.HIGHEST_PROTOCOL)

#plt.plot(z_marg_fit, label='Fit')
#plt.plot(margT/100, label='Data')
#plt.legend()
#plt.show()


#%% Spline 

#cs_avg = CubicSpline(w[:,0], avgT/100)
cs_avg = scipy.interpolate.BSpline(w[:,0], avgT/100, 5)

xa = np.linspace(0,2,1000)
plt.plot(100 * xa,avgT[:-1]/100, label='Data', linestyle='--')
plt.plot(100 * xa,cs_avg(xa), label='Fitted', linestyle='-')
plt.legend()
plt.xlabel('Pct. of Average Income')
plt.ylabel('Average Tax rate')
plt.gcf().set_size_inches(7/1.8, 5/1.8) 
plt.rcParams.update({'axes.titlesize': 'small'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.tight_layout()
plt.savefig('tax_function2.pdf')    
plt.show()   
       
    

cs_marg = scipy.interpolate.CubicSpline(w[:,0], margT/100)

plt.plot(margT/100, label='Data')
plt.plot(cs_marg(xa), label='Fit', linestyle='dashed')
plt.legend()
plt.show()

with open('cs_avg.pickle', 'wb') as f:
    pickle.dump(cs_avg, f, pickle.HIGHEST_PROTOCOL)

with open('cs_marg.pickle', 'wb') as f:
    pickle.dump(cs_marg, f, pickle.HIGHEST_PROTOCOL)



#%%

    x = np.linspace(0,3, num = 200)
    y =  z_avg[x]
    #y =  avgTaxf(x, 1)   
    plt.plot(x*100,y   , '-')       
    plt.xlabel('Pct. of Average Income')
    plt.ylabel('Average Tax rate')
    plt.gcf().set_size_inches(8, 5) 
    #plt.savefig('plots/tax_function.pdf')    
    plt.show()   
    
    
    
    
#%% Earnings risk

    
df = pd.read_excel (r'Male_earnings_betas.xlsx')
    
Elasticity = df[["y"]].to_numpy()   
perc       = df[["x"]].to_numpy()  
    

plt.plot(perc, Elasticity )
plt.show()

# Normalize
from scipy.optimize import curve_fit


def func(x, a, b, c, d):
    return a + b * x + c * x**2 - d * 2.7**x 

popt, pcov = curve_fit(func, perc[:,0], Elasticity[:,0])


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(perc[:,0], Elasticity, 'b-', label = 'Empirical Estimate')
ax.plot(perc[:,0], func(perc[:,0], *popt),  linestyle='--', label = 'Fit')

#ax.set_title('Assets')
ax.set_xlabel('Earnings percentile')
ax.set_ylabel('Elasticity of earnings to GDP')
plt.tight_layout()  
ax.legend(loc = 'best')
plt.savefig('Song_earnings_cyclical.pdf') 
plt.show()

np.save('popt', popt)

def func1(x, popt):
    return popt[0] + popt[1] * x + popt[2] * x**2 - popt[3] * 2.7**x 

#with open('Incidence.pkl','wb') as f:
#          pickle.dump(func1, f)
     
#Incidence = cloudpickle.dumps(func1)





