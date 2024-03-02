#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[2]:


#lets set the time (Duration of the data)
years = 15
endDate  = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)


# In[3]:


tickers = ['SPY','BND','GLD','QQQ','VTI']


# In[4]:


#creating the dataframe of the data we want for calculatine returns
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker,start = startDate, end = endDate)
    adj_close_df[ticker] = data['Adj Close']
print(adj_close_df)


# In[5]:


# daily log returns(These are net (r) not R)
# log return are easier for calculation
# this shift 1 is used to calculate the the (number /number-shifted 1 above) it

log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns = log_returns.dropna()

log_returns


# In[6]:


# we are asssuming that future returns are rely on past returns
def expected_return(weights,log_returns):
    return np.sum(log_returns.mean()*weights)


# In[7]:


#now we will create a portfolio standard deviation

def standard_deviation (weights , cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return(np.sqrt(variance))


# In[8]:


#create covariance matrix for all the assets
#252 trading days in a year

cov_matrix = log_returns.cov()
print(cov_matrix)


# In[9]:


# now we will create equally weighted portfolio

portfolio_value = 1000000
weights = np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = standard_deviation(weights,cov_matrix) 


# In[10]:


#create a function that gives a random Z-score based on normal distribution

def random_z_score():
    return np.random.normal(0,1)


# In[11]:


days = 5

def scenario_gain_loss(portfolio_value, portfolio_std_dev , z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value*portfolio_std_dev*z_score*np.sqrt(days)


# In[12]:


#Rum 10000 simulation
simulation = 10000
scenarioReturn = []

for i in range(simulation):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value,portfolio_std_dev,z_score,days))
    


# In[13]:


#confidence interval
confidence_interval = 0.99
VaR = np.percentile(scenarioReturn, 100*(1-confidence_interval))


# In[14]:


VaR


# In[15]:


#ploting this

plt.hist(scenarioReturn, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()


# In[ ]:




