""""""
"""OMSCS2022FALL-P8: Manual Strategy 		  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from util import get_data, plot_data
from matplotlib.dates import DateFormatter, MONDAY, MonthLocator, YearLocator
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import indicators as ind
import marketsimcode as mt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

def author():
    return 'jhan446'
"""
-------------------------------------------------------------------------------  		  	   		  	  		  		  		    	 		 		   		 		  
                            Manual Strategy
------------------------------------------------------------------------------- 		  	   		  	  		  		  		    	 		 		   		 		  
Parameters:
1. verbose (bool): If “verbose” is True, print out information; False, should not generate ANY output  		  	   		  	  		  		  		    	 		 		   		 		  
2. impact (float): The market impact of each transaction, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
3. commission (float): The commission amount charged, defaults to 0.0

Return:
df_trades (dataFrame): orders (long/short/do nothing)  	  		  		  		    	 		 		   		 		  
-------------------------------------------------------------------------------	  	   		  	  		  		  		    	 		 		   		 		  
"""
class ManualStrategy(object):

    def author(self):
        return 'jhan446'

    def __init__(self, verbose=False, impact=9.95, commission=0.005):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv = 100000):

        dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfStock[symbol].ffill().bfill()   # Filling missing date

        dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfSPY.join(dfStock)   # Drop the days when market is closed
        numD = dfStock.shape[0]

        position = np.empty(numD)
        trade = np.empty(numD)
        trade[-1] = 0   # No trades on the last day
        legalP = [-1000, 0, 1000]   # legal positions

        # Get indicators
        sd_ind = (sd - relativedelta(days=25)).strftime('%Y%m%d')
        sd_ind = dt.datetime.strptime(sd_ind, '%Y%m%d')
        psr = ind.PSRatio(symbol, sd_ind, ed, window=10)
        bbv = ind.BBV(symbol, sd_ind, ed, window=14)
        roc = ind.ROC(symbol, sd_ind, ed, window=9)

        dfStock['psr'] = psr
        dfStock['bbv'] = bbv
        dfStock['roc'] = roc

        trainX = pd.concat((dfStock['psr'], dfStock['bbv'], dfStock['roc']), axis=1)
        trainX = trainX.to_numpy()

        psr = dfStock['psr'].to_numpy()
        bbv = dfStock['bbv'].to_numpy()
        roc = dfStock['roc'].to_numpy()

        # Determine actions and trades
        action = np.zeros(numD)
        for i in range(numD - 1):
            if bbv[i] < -1:
                action[i] = 1
            elif bbv[i] > 1:
                action[i] = -1
            elif psr[i] < 1:
                action[i] = 1
            elif psr[i] > 1.105:
                action[i] = -1
            elif i > 0 and roc[i - 1] < 0 and roc[i] > 0:
                action[i] = 1
            elif i > 0 and roc[i - 1] > 0 and roc[i] < 0:
                action[i] = -1

            # Get position and trades
            if action[i] == -1:
                position[i] = legalP[0]
            elif action[i] == 1:
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
'''
-------------------------------------------------------------------------------
                            1. Calculate Statistics
-------------------------------------------------------------------------------
'''
def statistics(portvalsMS, portvalsBench, filename, verbose):

    fv = portvalsMS[-1]
    portvalsMS = portvalsMS / portvalsMS.iloc[0]
    cr = portvalsMS[-1] / portvalsMS[0] - 1
    dr = portvalsMS[1:] / portvalsMS[:-1].values - 1  # daily_return[0] does not exsit
    adr = dr.mean()
    sddr = dr.std()
    sr = math.sqrt(252) * (adr / sddr)

    fvB = portvalsBench[-1]
    portvalsBench = portvalsBench / portvalsBench.iloc[0]
    crB = portvalsBench[-1] / portvalsBench[0] - 1
    drB = portvalsBench[1:] / portvalsBench[:-1].values - 1  # daily_return[0] does not exsit
    adrB = drB.mean()
    sddrB = drB.std()
    srB = math.sqrt(252) * (adrB / sddrB)

    # Screen print for report
    if verbose:
        print()
        print(filename)
        print(f"Cumulative Return of MS and Benchmark:     {'%.6f' %cr}, {'%.6f' %crB}")
        print(f"Average Daily Return of MS and Benchmark:  {'%.6f' %adr}, {'%.6f' %adrB}")
        print(f"Sharpe Ratio of MS and Benchmark:          {'%.6f' %sr}, {'%.6f' %srB}")
        print(f"Final Portfolio Value of MS and Benchmark: {'%.1f' %fv}, {'%.1f' %fvB}")
        print(f"Standard Deviation of MS and Benchmark:    {'%.6f' % sddr}, {'%.6f' % sddrB}")
'''
-------------------------------------------------------------------------------
                            2. Plot Results
-------------------------------------------------------------------------------
'''
def plotResults(portvalsMS, portvalsBench, buy, sell, filename, verbose):

    # Get normalized portfolio value
    portvalsMS = portvalsMS / portvalsMS[0]
    portvalsBench = portvalsBench / portvalsBench[0]

    # Sort buy and sell
    numB = len(buy)
    numS = len(sell)

    # Screen print for report
    if verbose:
        print()
        print(filename)
        print('Number of buy: ', numB)
        print('Number of sell: ', numS)
        print('Total number of trades: ', numB + numS)

    # Plot
    fig, ax = plt.subplots(1, figsize=(9, 5))
    if filename == 'In-sample':
        first = [dt.date(2008,1,1).strftime('%Y/%m'), dt.date(2008,4,1).strftime('%Y/%m'),
                 dt.date(2008,7,1).strftime('%Y/%m'), dt.date(2008,10,1).strftime('%Y/%m'),
                 dt.date(2009,1,1).strftime('%Y/%m'), dt.date(2009,4,1).strftime('%Y/%m'),
                 dt.date(2009,7,1).strftime('%Y/%m'), dt.date(2009,10,1).strftime('%Y/%m'),
                 dt.date(2010,1,1).strftime('%Y/%m')]

        firsts = [dt.date(2008,1,1), dt.date(2008,4,1),
                 dt.date(2008,7,1), dt.date(2008,10,1),
                 dt.date(2009,1,1), dt.date(2009,4,1),
                 dt.date(2009,7,1), dt.date(2009,10,1),
                 dt.date(2010,1,1)]
        sd = dt.date(2008, 1, 1)
        ed = dt.date(2010, 1, 1)
        plt.ylim(0.6, 2.0)
        plt.yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    else:
        first = [dt.date(2010, 1, 1).strftime('%Y/%m'), dt.date(2010, 4, 1).strftime('%Y/%m'),
                 dt.date(2010, 7, 1).strftime('%Y/%m'), dt.date(2010, 10, 1).strftime('%Y/%m'),
                 dt.date(2011, 1, 1).strftime('%Y/%m'), dt.date(2011, 4, 1).strftime('%Y/%m'),
                 dt.date(2011, 7, 1).strftime('%Y/%m'), dt.date(2011, 10, 1).strftime('%Y/%m'),
                 dt.date(2012, 1, 1).strftime('%Y/%m')]

        firsts = [dt.date(2010, 1, 1), dt.date(2010, 4, 1),
                  dt.date(2010, 7, 1), dt.date(2010, 10, 1),
                  dt.date(2011, 1, 1), dt.date(2011, 4, 1),
                  dt.date(2011, 7, 1), dt.date(2011, 10, 1),
                  dt.date(2012, 1, 1)]
        sd = dt.date(2010, 1, 1)
        ed = dt.date(2012, 1, 1)
        plt.ylim(0.85, 1.1)
        plt.yticks([0.85, 0.9, 0.95, 1, 1.05, 1.1])

    mondays = MonthLocator()
    ax.xaxis.set_minor_locator(mondays)
#    ax.grid(color='gainsboro', linestyle='dashed', axis='x', which='minor')
    ax.grid(color='gainsboro', linestyle='dashed', axis='y', which='major')
    ax.tick_params(axis='x', direction='in', which='minor', length=4)
    ax.tick_params(axis='x', direction='out', which='major', labelsize=10, length=5)

    plt.xlim(sd, ed)
    plt.xticks(firsts, first, rotation=0)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title(filename + ' Manual Strategy vs. Benchmark with JPM')

    plt.plot(portvalsMS, 'r', label='Manual Strategy')
    plt.plot(portvalsBench, 'm', label='Benchmark')
    # Plot buy/sell
    for i in range(numB):
        ax.axvline(buy[i], c='b', linewidth=0.5)
    for i in range(numS):
        ax.axvline(sell[i], c='k', linewidth=0.5)

    plt.legend()
#    plt.grid(color='gainsboro', linestyle='dashed')
#    ax.text(0.5, 0.5, 'created by Jean', transform=ax.transAxes, fontsize=70, color='grey', alpha=0.5, ha='center', va='center', rotation=30)
    plt.savefig('MS-' + filename + '.png')
    plt.close()
'''
-------------------------------------------------------------------------------
                            3. Test Code
-------------------------------------------------------------------------------
'''
def test_code(verbose):
    ms = ManualStrategy()
    symbol = 'JPM'
    commission = 9.95
    impact = 0.005
    sv = 100000
    sd = [dt.datetime(2008, 1, 1), dt.datetime(2010, 1, 1)]
    ed = [dt.datetime(2009, 12, 31), dt.datetime(2011, 12, 31)]
    filename = ['In-sample', 'Out-of-sample']

    for i in range(2):
        df_trades = ms.testPolicy(symbol = symbol, sd  = sd[i], ed = ed[i], sv = sv)
        portvalsMS = mt.compute_portvals(df_trades, symbol, sv = sv, commission = commission, impact = impact)

        # Sort buy/sell trades
        buy = []
        sell = []
        id = -1
        for trade in df_trades:
            id = id + 1
            if trade > 0.0:
                buy.append(df_trades.index[id])
            elif trade < 0.0:
                sell.append(df_trades.index[id])

        df_trades = get_benchmark(symbol = symbol , sd = sd[i], ed = ed[i], sv = sv)
        portvalsBench = mt.compute_portvals(df_trades, symbol, sv = sv, commission = commission, impact = impact)

        statistics(portvalsMS, portvalsBench, filename[i], verbose)
        plotResults(portvalsMS, portvalsBench, buy, sell, filename[i], verbose)

if __name__ == "__main__":
    verbose = False
    test_code(verbose)
