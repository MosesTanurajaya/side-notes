# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:05:05 2019

@author: OwnerPC
"""
#===============================================================================
# LESSON 4.4
#===============================================================================

# Reading Data: Raw data as CSV  ---------------------------
with open('prices.csv', 'r') as file:
    prices = file.read() 
print(prices)

# Converting data to Dataframe -----------------------------
import pandas as pd
price_df = pd.read_csv('prices.csv')
price_df

# Giving Dataframe some descriptive (Header)
price_df = pd.read_csv('prices.csv', names=['ticker', 'date', 'open', 'high', 'low',
                                             'close', 'volume', 'adj_close', 
                                             'adj_volume'])
price_df

# Simple calculation with dataframe:
price_df.median()

# Grouping data on each ticker and summarize it as its median
price_df.groupby('ticker').median()

# Command for viewing specific row in row
price_df.iloc[[6, 7, 13, 14]]

# GROUPING by Pivoting Dataframe dimension and managing data presentation base upon its
# attributes, soalnya sebelumnya kecampur2
open_prices = price_df.pivot(index='date', columns='ticker', values='open')
high_prices = price_df.pivot(index='date', columns='ticker', values='high')
low_prices = price_df.pivot(index='date', columns='ticker', values='low')
close_prices = price_df.pivot(index='date', columns='ticker', values='close')
volume = price_df.pivot(index='date', columns='ticker', values='volume')
adj_close_prices = price_df.pivot(index='date', columns='ticker', values='adj_close')
adj_volume = price_df.pivot(index='date', columns='ticker', values='adj_volume')

open_prices # now we can select data based on its attributes ( more readable)

# Perform mathematical function to each of atttributes
open_prices.mean()

# TRANSPOSE to Choose about how to display the data. displaying mean 
# by each ticker or by each date
open_prices.T.mean()

import quiz_tests
import pandas as pd


def csv_to_close(csv_filepath, field_names):
    """Reads in data from a csv file and produces a DataFrame with close data.
    
    Parameters
    ----------
    csv_filepath : str
        The name of the csv file to read
    field_names : list of str
        The field names of the field in the csv file

    Returns
    -------
    close : DataFrame
        Close prices for each ticker and date
    """
    
    # TODO: Implement Function
    # return pd.read_csv(csv_filepath, names=field_names) # this gives dataframe
    # .pivot(index='date', columns='ticker', values='close') # This grouping of data by close price
    
    return pd.read_csv(csv_filepath, names=field_names).pivot(index='date', columns='ticker', values='close')


quiz_tests.test_csv_to_close(csv_to_close)

#===============================================================================
# LESSON 5.7
#===============================================================================

import numpy as np
import pandas as pd

dates = pd.date_range('10/10/2018', periods=11, freq='D') # create string of dates intended 
#use as index
close_prices = np.arange(len(dates))  # create what is inside dataframe

close = pd.Series(close_prices, dates) # Serialise those two columns of data

close.resample('3D') # from close, we boxing  the data into 3 row's each
close.resample('3D').first() # same from the box we display first value only
# Resample can only be used on time-related calss data

try:     # Attempt resample on a series without a time index  
    pd.Series(close_prices).resample('W')
except TypeError:
    print('It threw a TypeError.')
else:
    print('It worked.')
    # Failed, below is the correct one

pd.DataFrame({
    'days': close, # already has dates and value
    'weeks': close.resample('W').first()})
    
close.resample('W').ohlc()
"""
Resampler.count([_method])	Compute count of group, excluding missing values
Resampler.nunique([_method])	Returns number of unique elements in the group
Resampler.first([_method])	Compute first of group values
Resampler.last([_method])	Compute last of group values
Resampler.max([_method])	Compute max of group values
Resampler.mean([_method])	Compute mean of groups, excluding missing values
Resampler.median([_method])	Compute median of groups, excluding missing values
Resampler.min([_method])	Compute min of group values
Resampler.ohlc([_method])	Compute sum of values, excluding missing values
Resampler.prod([_method])	Compute prod of group values
Resampler.size()	Compute group sizes
Resampler.sem([_method])	Compute standard error of the mean of groups, excluding missing values
Resampler.std([ddof])	Compute standard deviation of groups, excluding missing values
Resampler.sum([_method])	Compute sum of group values
Resampler.var([ddof])	Compute variance of groups, excluding missing values
"""
    open_prices_weekly = open_prices.resample('W').first()
        #Weekly open prices for each ticker and date
    high_prices_weekly = high_prices.resample('W').max()
        #Weekly high prices for each ticker and date
    low_prices_weekly  = low_prices.resample('W').min()
        #Weekly low prices for each ticker and date
    close_prices_weekly = close_prices.resample('W').last()
        #Weekly close prices for each ticker and date
    
    # TODO: Implement Function
    
    return open_prices_weekly, high_prices_weekly , low_prices_weekly,  close_prices_weekly

#===============================================================================
# LESSON 7.2
#===============================================================================

import pandas as pd        # create dataframe
close = pd.DataFrame(
    {
        'ABC': [1, 5, 3, 6, 2],
        'EFG': [12, 51, 43, 56, 22],
        'XYZ': [35, 36, 36, 36, 37],},
    pd.date_range('10/01/2018', periods=5, freq='D'))
close
# push foward 2 days
close.shift(2)

#======================================================================
# LESSON 7.5  create distribution return and price
# Scenarion of stocks give returns 1% +- 2%, so its either -1% or +3%

from scipy.stats import bernoulli
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# this is the function to simulate random value generation
def generate_returns(num_returns):
    p = 0.5       # remember bernoulli always give 0 or 1, so this is where 0.5 came from
    return 0.01 + (bernoulli.rvs(p, size=num_returns)-0.5)*0.04 
print(generate_returns(6))

# This is the function to simulate n number of sample during N period of simulation
# 6+ 1 periode, initial value $ 100, sampling 1000x
# bins just create how many slots/ pembagian di dalam tampilan graph
final_values = [100*np.prod(generate_returns(6)+1) for i in range(1,1000)]
plt.hist(final_values, bins=100)
plt.ylabel('Frequency')
plt.xlabel('Value after 6 months')
plt.show()
# for 20 month simulation
final_values = [100*np.prod(generate_returns(20)+1) for i in range(1,1000)]
plt.hist(final_values, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Value after 20 months')
plt.show()
# REMEMBER: the bigger the sample the better, default value bin = 10, periode of simulation
# the longer the better to see TRUE data distribution

#=======================================================================
# LESSON 8.5 Dtype
#=======================================================================
import numpy as np

array = np.arange(10)
print(array)
print(type(array))
print(array.dtype)
# try to convert to float dtype
float_arr = array / 2
print(float_arr) # raw output
print(type(float_arr)) #print the class information
print(float_arr.dtype) #print literal dtype
# try to convert back to integer
int_arr = float_arr.astype(np.int64) # casting
print(int_arr)
print(type(int_arr))
print(int_arr.dtype)
# try to create trading signal example
prices = np.array([1, 3, -2, 9, 5, 7, 2])
prices
signal_one = prices > 2  # will produce boolean
signal_three = prices > 4
print(signal_one)
print(signal_three)
# from those boolean converet back as integer
signal_one = signal_one.astype(np.int)
signal_three = signal_three.astype(np.int)
print(signal_one)
print(signal_three)
# try to open postion absed on price input
pos_one = 1 * signal_one
pos_three = 3 * signal_three
print(pos_one)
print(pos_three)
#theoretical return based on input:
long_pos = pos_one + pos_three

print(long_pos)

#==========================================================================
#LESSON 8.8 try to find / sort data
#==========================================================================
import pandas as pd
month = pd.to_datetime('02/01/2018')
close_month = pd.DataFrame(
    {
        'A': 1,
        'B': 12,
        'C': 35,
        'D': 3,
        'E': 79,
        'F': 2,
        'G': 15,
        'H': 59},
    [month])
close_month
# Attempt to run nlargest
try:
    close_month.nlargest(2)
except TypeError as err:
    print('Error: {}'.format(err))
#What happeened here? It turns out we're not calling the Series.nlargest function, 
#we're actually calling DataFrame.nlargest, since close_month is a DataFrame. 
#Let's get the Series from the dataframe using .loc[month], 
#where month is the 2018-02-01 index created above.
close_month.loc[month].nlargest(2) # we only took one strip / colum of data and it is series

# to find bottom we can just flip sign or use n.smallest
(-1 * close_month).loc[month].nlargest(2) # trick by multiply -1 
(-1 * close_month).loc[month].nlargest(2) *-1 # reveal real prices after found
#QUIZ , this is to pick the n-top price for that day, and return the sector in it
import project_tests
import pandas as pd

def date_top_industries(prices, sector, date, top_n):
    """
    Get the set of the top industries for the date
    Parameters
    ----------
    prices : DataFrame
        Prices for each ticker and date
    sector : Series
        Sector name for each ticker
    date : Date
        Date to get the top performers
    top_n : int
        Number of top performers to get 
    Returns
    -------
    top_industries : set
        Top industries for the date
    """
    # TODO: Implement Function
    # fetch top prices at the specific date inputed
    # return as set
    #      #grab sector #at parameter price # this to find n-top number of prices
    return set(sector.loc[prices.loc[date].nlargest(top_n).index])
    
project_tests.test_date_top_industries(date_top_industries)

#============================================================================
# LESSON 8.11 t test Quiz
#============================================================================
import pandas as pd
import numpy as np
import scipy.stats as stats

def analyze_returns(net_returns):
    # trying to get series values net return parameter
    # Alternative 1:
    # net_returns = pd.core.series.Series(net_returns)
    net_returns = pd.Series(net_returns['return']) 
    null_hypothesis = 0
    #stats.ttest_1samp return 2 values
    t_value, p_value = stats.ttest_1samp(net_returns,null_hypothesis)
    return t_value, p_value/2

def test_run(filename='net_returns.csv'):

    # Alternative 1: 
    # net_returns = pd.Series.from_csv(filename, header=0) # first create the dataframe
    net_returns = pd.read_csv(filename, header=0)
    t, p = analyze_returns(net_returns)
    print("t-statistic: {:.3f}\np-value: {:.6f}".format(t, p))

if __name__ == '__main__':
    test_run()

#===========================================================================
#
#


































