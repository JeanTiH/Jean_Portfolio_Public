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
    dfStock['sma'] = sma
    dfStock['psr'] = psr
    return dfStock['sma'], dfStock['psr']

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
    dfStock['emaS'] = emaS
    dfStock['smaL'] = smaL
    dfStock['cross'] = cross
# Get local maximum/minimum
    crossG = cross.fillna(0.0).to_numpy()
    maxLocal = argrelextrema(crossG, np.greater)
    minLocal = argrelextrema(crossG, np.less)
    crossG[:] = 'Nan'
    crossG[maxLocal] = 1.0
    crossG[minLocal] = -1.0
    dfStock['crossG'] = crossG
    return dfStock['emaS'], dfStock['smaL'], dfStock['cross'], dfStock['crossG']

# Indicator 4 BB
def BBV(symbol, sd, ed, window):
    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    n = window

    sma = price.rolling(window = n).mean()
    std = price.rolling(window = n).std()
    bbv = (price - sma) / (std * 2.0)
    dfStock['sma'] = sma
    dfStock['upperB'] = sma + std * 2.0
    dfStock['lowerB'] = sma - std * 2.0
    dfStock['bbv'] = bbv
    return dfStock['sma'], dfStock['upperB'], dfStock['lowerB'], dfStock['bbv']

# Indicator 5 PPO
def PPO(symbol, sd, ed, window1, window2):
    dfStock = prepare_data(symbol, sd, ed)

    ema1 = EMA(symbol, sd, ed, window1, True)
    ema2 = EMA(symbol, sd, ed, window2, True)
    ppo = (ema1 - ema2) / ema2 * 100
    signal = EMA(ppo, sd, ed, 9, False)

    ppov = ppo - signal
    dfStock['ema1'] = ema1
    dfStock['ema2'] = ema2
    dfStock['ppo'] = ppo
    dfStock['signal'] = signal
    dfStock['ppov'] = ppov
    return dfStock['ema1'], dfStock['ema2'], dfStock['ppo'], dfStock['signal'], dfStock['ppov']
'''
-------------------------------------------------------------------------------
                            Plot Results of Indicators
-------------------------------------------------------------------------------
'''
def plotResults(price, window, ind_num, indicator, ind_name, helper, helper_name, helper_window):

    incident_num = 4
    incident_date = [dt.date(2008,5,1), dt.date(2008,7,14), dt.date(2009,3,9), dt.date(2009,10,12)]

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

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8,5), sharex = True)
# 1. Plot incident lines
    for i in range(incident_num):
        ax1.axvline(incident_date[i], c='g', linestyle='dashed', linewidth=1.5)
        ax2.axvline(incident_date[i], c='g', linestyle='dashed', linewidth=1.5)
# 2. Plot upper figure (price & helpers)
    # 2.1 Plot price
    ax1.plot(price, 'r', label='Price')
    # 2.2 Plot helpers
    if ind_num == 1:
        indicator_name = ind_name
    else:
        indicator_name = ind_name[0]

    if indicator_name == 'P/SMA' or indicator_name == 'ROC':
        ax1.plot(helper, 'k', label=helper_name + helper_window)
        ax1.legend(loc='lower right')
    elif indicator_name == 'Golden/Death Cross' or indicator_name == 'PPO':
        helper1 = helper[0]
        helper2 = helper[1]
        helper_name1 = helper_name[0]
        helper_name2 = helper_name[1]

        ax1.plot(helper1, 'k', label=helper_name1 + helper_window[0])
        ax1.plot(helper2, 'mediumspringgreen', label=helper_name2 + helper_window[1])
        ax1.legend(loc='lower right')
    elif indicator_name == 'Bollinger Bands':
        helper1 = helper[0]
        helper2 = helper[1]
        helper3 = helper[2]
        helper_name1 = helper_name[0]

        ax1.plot(helper1, 'k', label=helper_name1 + helper_window[0])
        ax1.plot(helper2, 'mediumspringgreen', label='Bollinger Bands')
        ax1.legend(loc='lower right')
        ax1.plot(helper3, 'mediumspringgreen')

    mondays = MonthLocator()
    ax1.xaxis.set_minor_locator(mondays)
    ax1.grid(color='gainsboro', linestyle='dashed',axis='x', which='minor')
    ax1.grid(color='gainsboro', linestyle='dashed',axis='y', which='major')
    ax1.set_ylabel('Normalized Value')
    ax1.set_ylim(0.2, 1.2)
    ax1.tick_params(axis="x", direction="in", which="minor", length=4)
    ax1.tick_params(axis="x", direction="out", which="major", labelsize=10, length=5)

    plt.xlim(dt.date(2008, 1, 1), dt.date(2010, 1, 1))
    plt.xticks(firsts, first, rotation=0)
    plt.suptitle(' Price and ' + indicator_name, size =15)
    plt.xlabel('Date')
# 3. Plot lower figure (indicators & threshold lines)
    # 3.1 Plot indicators
    if ind_num == 1:
        if indicator_name == 'Golden/Death Cross':
            indicator_name1 = 'G/D Cross Value'
        elif indicator_name == 'Bollinger Bands':
            indicator_name1 = 'BB Value'
        else:
            indicator_name1 = indicator_name
        ax2.plot(indicator, 'b', label=indicator_name1 + window)
    else:
        indicator_name1 = ind_name[0]
        ax2.plot(indicator[0], 'slateblue', label=ind_name[0] + window)
        ax2.plot(indicator[1], 'g', label=ind_name[1] + ' (window = 9)')
        ax2.bar(indicator[2].index, indicator[2], color='orange', label=ind_name[2] + window)
        ax2.axhline(0, c='r', linestyle='dashed', linewidth=1.5)

    ax2.set_ylabel(indicator_name1)
    ax2.legend(loc='lower right')
    ax2.grid(color='gainsboro', linestyle='dashed',axis='x', which='minor')
    ax2.grid(color='gainsboro', linestyle='dashed',axis='y', which='major')
    ax2.tick_params(axis='x', direction='in', which='minor', length=4)
    ax2.tick_params(axis='x', direction='out', which='major', labelsize=10, length=5)
    # 3.2 Add threshold lines
    if indicator_name == 'P/SMA':
        plt.ylim(0.6,1.4)
        ax2 = ax2.twinx()
        plt.ylim(0.6,1.4)
        plt.yticks([0.6,0.95,1.05,1.4], ['','0.95','1.05',''])
        ax2.axhline(0.95, c='r', linestyle='dashed', linewidth=1.5)
        ax2.axhline(1.05, c='r', linestyle='dashed', linewidth=1.5)
    elif indicator_name == 'ROC':
        plt.ylim(-60,60)
        ax2.axhline(0, c='r', linestyle='dashed', linewidth=1.5)
    elif indicator_name == 'Golden/Death Cross' or indicator_name == 'PPO':
        plt.ylim(-10,10)
        ax2.axhline(0, c='r', linestyle='dashed', linewidth=1.5)
    elif indicator_name == 'Bollinger Bands':
        plt.ylim(-2,2)
        ax2.axhline(-1, c='r', linestyle='dashed', linewidth=1.5)
        ax2.axhline(1, c='r', linestyle='dashed', linewidth=1.5)

#    plt.legend()
    indicator_name = indicator_name.replace('/', '_')
    plt.savefig(indicator_name + '.png')
    plt.close()
'''
-------------------------------------------------------------------------------
                                    3. Test Code
-------------------------------------------------------------------------------
'''
def test_code():
# 1. Price data
    symbol = 'JPM'
    sd = dt.datetime(2007, 10, 10)
    ed = dt.datetime(2009, 12, 31)

    dfStock = prepare_data(symbol, sd, ed)
    price = dfStock[symbol]
    price = price/price[0]
#----------------------------
# 2.1 Indicator1 - P/SMA Ratio
    window = 20
    w1 = ' (window = ' + str(window) + ')'
    sma, psr = PSRatio(symbol, sd, ed, window)
    psrN = 'P/SMA'
# Helper - sma
    sma = sma / abs(sma.dropna()[0])
    smaN = 'SMA'
#----------------------------
# 2.2 Indicator2 - ROC
    window = 9
    w2 = ' (window = ' + str(window) + ')'
    roc = ROC(symbol, sd, ed, window)
    rocN = 'ROC'
# Helper - sma
    ema = EMA(symbol, sd, ed, window, True)
    ema = ema / abs(ema.dropna()[0])
    emaN = 'EMA'
# ----------------------------
# 2.3 Indicator3 - Golden/Death Cross
    w3 = ''
    window_emaS = 15
    window_smaL = 50
    w_emaS = ' (window = ' + str(window_emaS) + ')'
    w_smaL = ' (window = ' + str(window_smaL) + ')'
    emaS, smaL, cross, crossG = Cross(symbol, sd, ed, windowS = window_emaS, windowL = window_smaL)
#    print(crossG)
    crossN = 'Golden/Death Cross'
# Helper - sma & ema
    emaS = emaS / abs(emaS.dropna()[0])
    smaL = smaL / abs(smaL.dropna()[0])
    emaSN = 'EMS'
    smaLN = 'SMA'
# ----------------------------
# 2.4 Indicator4 - Bollinger Bands
    window = 20
    w4 = ' (window = ' + str(window) + ')'
    smaB, upperB, lowerB, bbv = BBV(symbol, sd, ed, window)
    bbvN = 'Bollinger Bands'
# Helper - sma & upper/lower band
    smaB = smaB / abs(smaB.dropna()[0])
    upperB = upperB / abs(upperB.dropna()[0])
    lowerB = lowerB / abs(lowerB.dropna()[0])
    smaBN = 'SMA'
    upperBN = 'Upper Band'
    lowerBN = 'Lower Band'
# ----------------------------
# 2.5 Indicator5 - Percentage Price Oscillator
    w5 = ''
    window_ema1 = 12
    window_ema2 = 26
    w_ema1 = ' (window = ' + str(window_ema1) + ')'
    w_ema2 = ' (window = ' + str(window_ema2) + ')'
    ema1, ema2, ppo, signal, ppov = PPO(symbol, sd, ed, window1 = window_ema1, window2 = window_ema2)
    ppoN = 'PPO'
    signalN = 'signal'
    ppovN = 'PPO Difference'
# Helper - ema
    ema1 = ema1 / abs(ema1.dropna()[0])
    ema2 = ema2 / abs(ema2.dropna()[0])
    ema1N = 'EMA'
    ema2N = 'EMA'
# ----------------------------
    plotResults(price, w1, 1, psr, psrN, sma, smaN, w1)
    plotResults(price, w2, 1, roc, rocN, ema, emaN, w2)
    plotResults(price, w3, 1, cross, crossN, (emaS, smaL), (emaSN, smaLN), (w_emaS, w_smaL))
    plotResults(price, w4, 1, bbv, bbvN, (smaB, upperB, lowerB), (smaBN, upperBN, lowerBN), (w4, w4, w4))
    plotResults(price, w5, 3, (ppo,signal,ppov), (ppoN,signalN,ppovN), (ema1,ema2), (ema1N,ema2N), (w_ema1,w_ema2))

if __name__ == "__main__":
    test_code()