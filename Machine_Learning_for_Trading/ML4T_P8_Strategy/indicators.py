""""""
"""OMSCS2022FALL-P8: Manual Strategy 		  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import numpy as np
import pandas as pd
from util import get_data, plot_data
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MONDAY, MonthLocator, YearLocator
from scipy.signal import argrelextrema
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
'''
-------------------------------------------------------------------------------
'''
def author():
    return 'jhan446'
'''
-------------------------------------------------------------------------------
                                Technical Indicators
-------------------------------------------------------------------------------
1. P/S Ratio (Price/Simple Moving Average) 
2. ROC (Price Rate of Change)
3. Cross (Golden and Death Cross)
4. BBV (Bollinger Bands)
5. PPO (Percentage Price Oscillator)
Note: P/SMA and SMA (BB and BBPercentage ) are considered to be equivalent in this project
-------------------------------------------------------------------------------
'''
# Prepare input stock data
def prepare_data(symbol, sd, ed):
    dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfStock[symbol].ffill().bfill()   # Filling missing date

    dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfSPY.join(dfStock)   # Drop the days when market is closed
    return dfStock

# Calculate SMA/EMA for indicators
def SMA(symbol, sd, ed, window):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    n = window

    sma = price.rolling(window = n).mean()
    return sma

def EMA(symbol, sd, ed, window, read_file):
    if read_file == True:   # Ture means calculate EMA of price
        dfStock = prepare_data(symbol, sd, ed)
        price = dfStock[symbol]
    else:                   # Fasle means calculate EMA of other index, such as EMA of PPO
        price = symbol
    n = window

    a = 2.0 / (n + 1.0)
    ema = price.ewm(alpha=a, min_periods=n, adjust=False).mean()
    return ema

# Indicator 1 P/SMA
def PSRatio(symbol, sd, ed, window):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    n = window

    sma = price.rolling(window = n).mean()
    psr = price / sma
    dfStock['psr'] = psr
    return dfStock['psr']

# Indicator 2 ROC
def ROC(symbol, sd, ed, window):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    n = window

    roc = (price[n:] - price[: -n].values)/ price[:-n].values * 100.0
    dfStock['roc'] = roc
    return dfStock['roc']

# Indicator 3 G/D Cross
def Cross(symbol, sd, ed, windowS, windowL):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]

    emaS = EMA(symbol, sd, ed, windowS, True)
    smaL = price.rolling(window = windowL).mean()
    cross = emaS - smaL
# Get local maximum/minimum
    crossG = cross.fillna(0.0).to_numpy()
    maxLocal = argrelextrema(crossG, np.greater)
    minLocal = argrelextrema(crossG, np.less)
    crossG[:] = 0.0
    crossG[maxLocal] = 1.0
    crossG[minLocal] = -1.0
    dfStock['crossG'] = crossG
    return dfStock['crossG']

# Indicator 4 BB
def BBV(symbol, sd, ed, window):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    n = window

    sma = price.rolling(window = n).mean()
    std = price.rolling(window = n).std()
    bbv = (price - sma) / (std * 2.0)
    dfStock['bbv'] = bbv
    return dfStock['bbv']

# Indicator 5 PPO
def PPO(symbol, sd, ed, window1, window2):
    dfStock = prepare_data(symbol, sd, ed)

    ema1 = EMA(symbol, sd, ed, window1, True)
    ema2 = EMA(symbol, sd, ed, window2, True)
    ppo = (ema1 - ema2) / ema2 * 100
    signal = EMA(ppo, sd, ed, 9, False)

    ppov = ppo - signal
    dfStock['ppov'] = ppov
    return dfStock['ppov']