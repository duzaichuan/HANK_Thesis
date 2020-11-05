# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:23:40 2020

@author: Nicolai
"""
import os
import sys


os.chdir(os.path.dirname(os.path.abspath(__file__)))


import pickle
import statsmodels.api as sm
import pandas as pd
import numpy as np

from fredapi import Fred

from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('ggplot')
#plt.style.use('bmh')
%config InlineBackend.figure_format = 'retina'

#%config InlineBackend.figure_format = 'retina'


params = {'legend.fontsize': 'small',
          'figure.figsize': (7, 5),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small',
         'figure.frameon' : 'True',
         'axes.edgecolor' : 'black'}



#%%

with open('data/data_fred.pickle', 'rb') as handle:
    data_fred = pickle.load(handle)
with open('data/data_rb.pickle', 'rb') as handle:
    data_rb = pickle.load(handle)
    
varspecs = [('u','UNRATE'),
          ('GDP','GDP'),
          ('deflator','GDPDEF'),
          ('pop','CNP16OV'),
          ('emp','CE16OV'),
          ('productivity_nonfarm', 'PRS85006163'),
          ('unemp','UNEMPLOY'),
          ('unemp_short','UEMPLT5'),
          ('wage_tot','A132RC1Q027SBEA'),
          ('hours','B4701C0A222NBEA'),
          ('R_qtr','BOGZ1FL072052006Q'),
          ('CPI','CPIAUCSL')]
            
# c. data variables
varnames = ['u_qtr','vt_qtr','lambda_u_qtr','deltat_qtr','y_qtr','w_qtr','a_qtr','R_qtr','Pi_qtr','Pi_CPI_qtr']  

data1 = {}
years = data_rb.index.year     
I = (years >= 1951) & (years <= 2003)
data1['vt'] = data_rb[f'vt'][I].values


data = {}
for varname,_varname_fred in varspecs:
    years = data_fred[f'{varname}'].index.year
    I = (years >= 1951) & (years <= 2003)
    data[varname] = data_fred[f'{varname}'][I].values

data  = {**data_rb, **data_fred}

#%%

u = data['u'].to_frame()
#u.Date = pd.to_datetime(u.index)
#u.set_index('Date', inplace=True)
u_qtr = u.resample('QS').mean()
#hej.index = hej.index.dt.to_period('Q')

vac_qtr = data_rb.resample('QS').mean()

years = vac_qtr.index.year
I_vt_old = (years >= 1951) & (years <= 1968)
I_vt_mid = (years >= 1969) & (years <= 1986)
I_vt_new = (years >= 1987) & (years <= 2003)

years = u_qtr.index.year
I_u_old = (years >= 1951) & (years <= 1968)
I_u_mid = (years >= 1969) & (years <= 1986)
I_u_new = (years >= 1987) & (years <= 2003)

list1 = [I_vt_old, I_vt_mid, I_vt_new]
list2 = [I_u_old, I_u_mid, I_u_new]
list3 = ['1951-1968', '1969-1986', '1987-2003']

colors = ['firebrick', 'darkgreen', 'dodgerblue']

ax = plt.subplot(1,1,1)


for k in range(3):
    y = vac_qtr[list1[k]]['vt']
    x = u_qtr[list2[k]][0]
      
    b, m = np.polynomial.polynomial.polyfit(x, y, 1)
    
    ax.plot(x, y,  linestyle='', marker='o', label = list3[k], color = colors[k])
    ax.plot(x, b + m * x, '-', color = colors[k])
    
pylab.rcParams.update(params)
plt.gcf().set_size_inches(7/1.2, 5/1.2) 
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
ax.legend(loc="best")

ax.set_xlabel('Unemployment Rate')
ax.set_ylabel('Vacancy Rate')
pylab.rcParams.update(params)
   
plt.tight_layout()
plt.savefig('plots/Beveridge.pdf')  
plt.show()




#%%
u['date'] = u.index.astype('datetime64[ns]')
u['date'] = pd.to_datetime(u["date"].dt.strftime('%Y-%m')).dt.month

#%%
u.reset_index(drop=True, inplace=True)

u.groupby(pd.PeriodIndex(u.columns, freq='Q'), axis=1).mean()



#%%
data  = {**data_rb, **data_fred}
for j in ['u','pop','vt','CPI']:   
    data[j].to_frame()
    #data[j] = pd.to_datetime(data[j]) 
    #data[j].index = data[j].index.to_period("Q")
    #data[j]['quarter'] = data[j].Date.dt.quarter
    #data[j].groupby(data[j].index).agg('sum')
    data[j].groupby(pd.PeriodIndex(data[j].index, freq='Q')).mean()
    

#%%
years = data_fred[f'unemp'].index.year
I = (years >= 1951) & (years <= 2003)
J = years[I] >= 1993
data['unemp_short'][J] *= 1.1 # adjustment suggested by Shimer (2005)
data['unemp'][J] *= 1.1 # adjustment suggested by Shimer (2005)

# b. formulas
data['lambda_u'] = np.nan*np.ones(data['unemp'].size)
data['deltat'] = np.nan*np.ones(data['unemp'].size)
data['lambda_u'][:-1] = 1.0-(data['unemp'][1:]-data['unemp_short'][1:])/data['unemp'][:-1]
data['deltat'][:-1] = data['unemp_short'][1:]/(data['emp'][:-1]*(1-0.5*data['lambda_u'][:-1]))


#%%

df_sales = data_fred
df_sales.date = pd.to_datetime(df_sales.date)

(df_sales.groupby(pd.PeriodIndex(df_sales.columns, freq='Q'), axis=1).apply(lambda x: x.sum(axis=1)/x.shape[1]))


 


#%%

def m_to_qrt(data, method):
    
    size = round(data_rb['vt'].size/3)
    output = np.empty()
    if method == 'sum':
        
    return output


data_rb['Date'] = pd.Series.to_datetime(data_rb['date'])
df['quarter'] = data_rb.Date.dt.quarter
vacancies =  data_rb.groupby(['quarter'])['vt'].mean()

#%%
# Sample data
x = np.arange(10)
y = 5 * x + 10

# Fit with polyfit
b, m = np.polynomial.polynomial.polyfit(x, y, 1)

plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()
