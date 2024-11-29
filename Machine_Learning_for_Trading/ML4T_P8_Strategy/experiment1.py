""""""
"""OMSCS2022FALL-P8: Experiment 1 		  	   		  	  		  		  		    	 		 		   		 		  

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

import marketsimcode as mt
import ManualStrategy as ms
import StrategyLearner as sl

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

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
def author():
    return 'jhan446'
def gtid():
    return 903845311
'''
-------------------------------------------------------------------------------
                            1. Calculate Statistics
-------------------------------------------------------------------------------
'''
def statistics(portvalsMS, portvalsBench, portvalsSL, filename, verbose):

    fvMS = portvalsMS[-1]
    portvalsMS = portvalsMS / portvalsMS.iloc[0]
    crMS = portvalsMS[-1] / portvalsMS[0] - 1
    drMS = portvalsMS[1:] / portvalsMS[:-1].values - 1  # daily_return[0] does not exsit
    adrMS = drMS.mean()
    sddrMS = drMS.std()
    srMS = math.sqrt(252) * (adrMS / sddrMS)

    fvB = portvalsBench[-1]
    portvalsBench = portvalsBench / portvalsBench.iloc[0]
    crB = portvalsBench[-1] / portvalsBench[0] - 1
    drB = portvalsBench[1:] / portvalsBench[:-1].values - 1  # daily_return[0] does not exsit
    adrB = drB.mean()
    sddrB = drB.std()
    srB = math.sqrt(252) * (adrB / sddrB)

    fvSL = portvalsSL[-1]
    portvalsSL = portvalsSL / portvalsSL.iloc[0]
    crSL = portvalsSL[-1] / portvalsSL[0] - 1
    drSL = portvalsSL[1:] / portvalsSL[:-1].values - 1  # daily_return[0] does not exsit
    adrSL = drSL.mean()
    sddrSL = drSL.std()
    srSL = math.sqrt(252) * (adrSL / sddrSL)

    if verbose:
        print()
        print(filename)
        print(f"Cumulative Return of SL, MS, Benchmark:     {'%.6f' %crSL}, {'%.6f' %crMS}, {'%.6f' %crB}")
        print(f"Average Daily Return of SL, MS, Benchmark:  {'%.6f' %adrSL}, {'%.6f' %adrMS}, {'%.6f' %adrB}")
        print(f"Sharpe Ratio of SL, MS, Benchmark:          {'%.6f' %srSL}, {'%.6f' %srMS}, {'%.6f' %srB}")
        print(f"Final Portfolio Value of SL, MS, Benchmark: {'%.1f' %fvSL}, {'%.1f' %fvMS}, {'%.1f' %fvB}")
        print(f"Standard Deviation of SL, MS, Benchmark:    {'%.6f' % sddrSL}, {'%.6f' % sddrMS}, {'%.6f' % sddrB}")
'''
-------------------------------------------------------------------------------
                            2. Plot Results
-------------------------------------------------------------------------------
'''
def plotResults(portvalsMS, portvalsBench, portvalsSL, filename):

    # Get normalized portfolio value
    portvalsMS = portvalsMS / portvalsMS[0]
    portvalsSL = portvalsSL / portvalsSL[0]
    portvalsBench = portvalsBench / portvalsBench[0]

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
        plt.ylim(0.5, 5)
        plt.yticks([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
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
        plt.ylim(0.8, 1.1)
        plt.yticks([0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1])

    plt.xlim(sd, ed)
    plt.xticks(firsts, first, rotation=0)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title(filename + ' Manual Strategy vs. Benchmark vs. Strategy Learner with JPM')

    plt.plot(portvalsMS, 'r', label='Manual Strategy')
    plt.plot(portvalsBench, 'm', label='Benchmark')
    plt.plot(portvalsSL, 'b', label='Strategy Learner')

    mondays = MonthLocator()
    ax.xaxis.set_minor_locator(mondays)
    ax.grid(color='gainsboro', linestyle='dashed', axis='x', which='minor')
    ax.grid(color='gainsboro', linestyle='dashed', axis='y', which='major')
    ax.tick_params(axis='x', direction='in', which='minor', length=4)
    ax.tick_params(axis='x', direction='out', which='major', labelsize=10, length=5)

    plt.legend()
    plt.grid(color='gainsboro', linestyle='dashed')
#    ax.text(0.5, 0.5, 'created by Jean', transform=ax.transAxes, fontsize=70, color='grey', alpha=0.5, ha='center', va='center', rotation=30)
    plt.savefig('EXP1-' + filename + '.png')
    plt.close()
'''
-------------------------------------------------------------------------------
                                3. Test Code 
                    i = 0 in-sample; i = 1 out-of-sample
-------------------------------------------------------------------------------
'''
def test_code(verbose):
    symbol = 'JPM'
    commission = 9.95
    impact = 0.005
    sv = 100000
    sd = [dt.datetime(2008, 1, 1), dt.datetime(2010, 1, 1)]
    ed = [dt.datetime(2009, 12, 31), dt.datetime(2011, 12, 31)]
    filename = ['In-sample', 'Out-of-sample']

    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd[0], ed=ed[0], sv=sv)  # Train the learner
    for i in range(2):
        df_trades = ms.ManualStrategy().testPolicy(symbol = symbol, sd  = sd[i], ed = ed[i], sv = sv)
        portvalsMS = mt.compute_portvals(df_trades, symbol, sv = sv, commission = commission, impact = impact)

        df_trades = ms.get_benchmark(symbol = symbol , sd = sd[i], ed = ed[i], sv = sv)
        portvalsBench = mt.compute_portvals(df_trades, symbol, sv = sv, commission = commission, impact = impact)

        df_trades = learner.testPolicy(symbol=symbol, sd=sd[i], ed=ed[i], sv=sv)
        portvalsSL = mt.compute_portvals(df_trades, symbol, sv = sv, commission = commission, impact = impact)

        # Count buy/sell trades
        numb = 0
        nums = 0
        for trade in df_trades.values:
            if trade > 0.0:
                numb = numb + 1
            elif trade < 0.0:
                nums = nums + 1
        num = numb + nums

        if verbose:
            print()
            print('--------------')
            print('Exp1')
            print('--------------')
            print(filename[i])
            print('numB= ', numb)
            print('numS= ', nums)
            print('numT= ', num)

        statistics(portvalsMS, portvalsBench, portvalsSL, filename[i], verbose)
        plotResults(portvalsMS, portvalsBench, portvalsSL, filename[i])

if __name__ == "__main__":
    np.random.seed(gtid())
    verbose = False
    test_code(verbose)
