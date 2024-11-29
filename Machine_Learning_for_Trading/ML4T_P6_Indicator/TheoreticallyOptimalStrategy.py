""""""
"""OMSCS2022FALL-P6: Theoretically Optimal Strategy	  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446 		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""
import numpy as np
import pandas as pd
from util import get_data, plot_data
import datetime as dt
'''
-------------------------------------------------------------------------------
'''
def author():
    return 'jhan446'
'''
-------------------------------------------------------------------------------
                    Theoretically Optimal Strategy (TOP)
2008-01-01 to 2009-12-31 JPM, allowable positions are 1000 shares long, 1000 shares short, 0 shares
You may trade up to 2000 shares at a time as long as your positions are 1000 shares long or 1000 shares short or 0
Commission = $0.00
Impact = 0.00

Return:
df_trades (pandas.DataFrame): A single-column dataframe, containing the action on each trading day
-------------------------------------------------------------------------------
'''
def testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv = 100000):

    dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfStock[symbol].ffill().bfill()   # Filling missing date

    dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfSPY.join(dfStock)   # Drop the days when market is closed
    numD = dfStock.shape[0]

    position = np.empty(numD)
    trade = np.empty(numD)
    trade[-1] = 0   # No trades on the last day
    legalP = [-1000, 0, 1000]   # legal positions
    for i in range(numD - 1):
        if dfStock[symbol][i + 1] < dfStock[symbol][i]:
            position[i] = legalP[0]
        elif dfStock[symbol][i + 1] > dfStock[symbol][i]:
            position[i] = legalP[2]
        else:
            if i == 0:
                position[i] = legalP[1]
            else:
                position[i] = position[i-1]

        if i == 0:
            trade[i] = position[i]
        else:
            trade[i] = position[i] - position[i - 1]

    dfStock['Trade'] = trade
    df_trades = dfStock['Trade']

    return df_trades
'''
-------------------------------------------------------------------------------
                                    Benchmark
2008-01-01 to 2009-12-31 JPM, starting with $100,000 cash, investing in 1000 shares of JPM, and holding that position

Return:
df_trades (pandas.DataFrame): A single-column dataframe, containing the action on each trading day
-------------------------------------------------------------------------------
'''
def get_benchmark(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv = 100000):

    dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfStock[symbol].ffill().bfill()   # Filling missing date

    dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfSPY.join(dfStock)   # Drop the days when market is closed
    numD = dfStock.shape[0]

    trade = np.zeros(numD)
    trade[0] = 1000
    dfStock['Trade'] = trade
    df_trades = dfStock['Trade']

    return df_trades