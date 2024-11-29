""""""
"""OMSCS2022FALL-P8: Experiment 2 		  	   		  	  		  		  		    	 		 		   		 		  

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
def statistics(portvalsSL, filename, imp, verbose):

    fvSL1 = portvalsSL[0][-1]
    portvalsSL1 = portvalsSL[0] / portvalsSL[0].iloc[0]
    crSL1 = portvalsSL1[-1] / portvalsSL1[0] - 1
    drSL1 = portvalsSL1[1:] / portvalsSL1[:-1].values - 1  # daily_return[0] does not exsit
    adrSL1 = drSL1.mean()
    sddrSL1 = drSL1.std()
    srSL1 = math.sqrt(252) * (adrSL1 / sddrSL1)

    fvSL2 = portvalsSL[1][-1]
    portvalsSL2 = portvalsSL[1] / portvalsSL[1].iloc[0]
    crSL2 = portvalsSL2[-1] / portvalsSL2[0] - 1
    drSL2 = portvalsSL2[1:] / portvalsSL2[:-1].values - 1  # daily_return[0] does not exsit
    adrSL2 = drSL2.mean()
    sddrSL2 = drSL2.std()
    srSL2 = math.sqrt(252) * (adrSL2 / sddrSL2)

    fvSL3 = portvalsSL[2][-1]
    portvalsSL3 = portvalsSL[2] / portvalsSL[2].iloc[0]
    crSL3 = portvalsSL3[-1] / portvalsSL3[0] - 1
    drSL3 = portvalsSL3[1:] / portvalsSL3[:-1].values - 1  # daily_return[0] does not exsit
    adrSL3 = drSL3.mean()
    sddrSL3 = drSL3.std()
    srSL3 = math.sqrt(252) * (adrSL3 / sddrSL3)

    # Screen print for report
    if verbose:
        print()
        print(filename)
        print("Cumulative Return of     " + imp + f": {'%.6f' %crSL1}, {'%.6f' %crSL2}, {'%.6f' %crSL3}")
        print("Average Daily Return of  " + imp + f": {'%.6f' %adrSL1}, {'%.6f' %adrSL2}, {'%.6f' %adrSL3}")
        print("Standard Deviation of    " + imp + f": {'%.6f' %sddrSL1}, {'%.6f' %sddrSL2}, {'%.6f' %sddrSL3}")
        print("Sharpe Ratio of          " + imp + f": {'%.6f' %srSL1}, {'%.6f' %srSL2}, {'%.6f' %srSL3}")
        print("Final Portfolio Value of " + imp + f": {'%.1f' %fvSL1}, {'%.1f' %fvSL2}, {'%.1f' %fvSL3}")
'''
-------------------------------------------------------------------------------
                            2. Plot Metric
-------------------------------------------------------------------------------
'''
def plotResults(numB, numS, portvalsSL, imp):

    sr = []
    cr = []
    # Get normalized portfolio value, sr, cr
    for i in range(3):
        portvalsSL[i] = portvalsSL[i] / portvalsSL[i][0]

        dr = (portvalsSL[i][1:] / portvalsSL[i][:-1].values - 1)
        adr = dr.mean()
        sddr = dr.std()
        sr.append(math.sqrt(252) * (adr / sddr))
        cr.append(portvalsSL[i][-1] / portvalsSL[i][0] - 1)
    '''
    Plot Metric 1
    '''
    fig, ax = plt.subplots(1, figsize=(9, 5))
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

    mondays = MonthLocator()
    ax.xaxis.set_minor_locator(mondays)
    ax.grid(color='gainsboro', linestyle='dashed', axis='x', which='minor')
    ax.grid(color='gainsboro', linestyle='dashed', axis='y', which='major')
    ax.tick_params(axis='x', direction='in', which='minor', length=4)
    ax.tick_params(axis='x', direction='out', which='major', labelsize=10, length=5)

    plt.ylim(0.5, 4.5)
    plt.yticks([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    plt.xlim(sd, ed)
    plt.xticks(firsts, first, rotation=0)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Strategy Learner In-sample Performance with Different Impact Values')

    plt.plot(portvalsSL[0], 'r', label=imp[0])
    plt.plot(portvalsSL[1], 'm', label=imp[1])
    plt.plot(portvalsSL[2], 'b', label=imp[2])

    plt.legend()
    plt.grid(color='gainsboro', linestyle='dashed')
#    ax.text(0.5, 0.5, 'created by Jean', transform=ax.transAxes, fontsize=70, color='grey', alpha=0.5, ha='center', va='center', rotation=30)
    plt.savefig('EXP2-Metric1.png')
    plt.close()
    '''
    Plot Metric 2
    '''
    fig, ax = plt.subplots(1, figsize=(9, 5))
    barWidth = 0.2
    N = len(numB)
    plt.xticks([r + barWidth for r in range(N)],[imp[0], imp[1], imp[2]])
    bar1 = np.arange(N)
    bar2 = [x + barWidth for x in bar1]
    bar3 = [x + barWidth for x in bar2]

    plt.ylim(0, 220)
    plt.yticks(np.arange(0, 221, 20))
    plt.xlabel('Impact')
    plt.ylabel('Number of Trades')
    plt.title('Strategy Learner In-sample Performance with Different Impact Values')
    p1 = plt.bar(bar1, numB, width=barWidth, color='b')
    p2 = plt.bar(bar1, numS, bottom=numB, width=barWidth, color='r')

    ax.twinx()
    plt.ylim(0, 7.001)
    plt.yticks(np.arange(0, 7.001, 0.5))
    plt.ylabel('Sharpe Ratio & Cumulative Return Value')
    p3 = plt.bar(bar2, sr, width=barWidth, color='g')
    p4 = plt.bar(bar3, cr, width=barWidth, color='k')

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Long', 'Short','Sharpe Ratio', 'Cumulative Return'), loc=1)
    plt.grid(axis='y', color='gainsboro', linestyle='dashed')
    plt.savefig('EXP2-Metric2.png')
    plt.close()
'''
-------------------------------------------------------------------------------
                            3. Test Code (in-sample)
-------------------------------------------------------------------------------
'''
def test_code(verbose):
    symbol = 'JPM'
    commission = 0.0
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    filename = 'In-sample'
    imp_index = [0.0, 0.008, 0.01]
    imp_plot = []
    portvalsSL = []
    imp_stat = 'impact = '
    numB = []
    numS = []
    numT = []

    for i in range(3):
        impact = imp_index[i]
        imp_plot.append('impact = ' + str(impact))
        if i == 2:
            imp_stat = imp_stat + str(impact)
        else:
            imp_stat = imp_stat + str(impact) + ', '
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)  # Train the learner
        df_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        portvalsSL.append(mt.compute_portvals(df_trades, symbol, sv=sv, commission=commission, impact=impact))

        # Count buy/sell trades
        numb = 0
        nums = 0
        for trade in df_trades.values:
            if trade > 0.0:
                numb = numb + 1
            elif trade < 0.0:
                nums = nums + 1
        num = numb + nums
        numB.append(numb)
        numS.append(nums)
        numT.append(num)

    # Screen print for report
    if verbose:
        print()
        print('--------------')
        print('Exp2')
        print('--------------')
        print('imp= ', imp_index)
        print('numB= ', numB)
        print('numS= ', numS)
        print('numT= ', numT)

    statistics(portvalsSL, filename, imp_stat, verbose)
    plotResults(numB, numS, portvalsSL, imp_plot)

if __name__ == "__main__":
    np.random.seed(gtid())
    verbose = False
    test_code(verbose)
