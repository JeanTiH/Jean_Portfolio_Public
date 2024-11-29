""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import matplotlib.pyplot as plt

import RTLearner as rt
import BagLearner as bl
import indicators as ind

def author():
    return 'jhan446'
"""
-------------------------------------------------------------------------------  		  	   		  	  		  		  		    	 		 		   		 		  
                            Strategy Learner
    learn a trading policy using the same indicators used in ManualStrategy
------------------------------------------------------------------------------- 		  	   		  	  		  		  		    	 		 		   		 		  
Parameters:
1. verbose (bool): If “verbose” is True, print out information; False, should not generate ANY output  		  	   		  	  		  		  		    	 		 		   		 		  
2. impact (float): The market impact of each transaction, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
3. commission (float): The commission amount charged, defaults to 0.0

Return:
df_trades (dataFrame): orders (long/short/do nothing)  		  	   		  	  		  		  		    	 		 		   		 		  
-------------------------------------------------------------------------------	  	   		  	  		  		  		    	 		 		   		 		  
"""
class StrategyLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  

    def author(self):
        return 'jhan446'

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  	  		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5},bags=20,boost=False,verbose=False)

    # this method creates a RTLearner, and train it for trading
    def add_evidence(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):

        dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfStock[symbol].ffill().bfill()  # Filling missing date

        dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfSPY.join(dfStock)  # Drop the days when market is closed
        price = dfStock[symbol]
        numD = dfStock.shape[0]
        # Get indicators
        sd_ind = (sd - relativedelta(days=25)).strftime('%Y%m%d')
        sd_ind = dt.datetime.strptime(sd_ind, '%Y%m%d')
        psr = ind.PSRatio(symbol, sd_ind, ed, window=10)
        bbv = ind.BBV(symbol, sd_ind, ed, window=14)
        roc = ind.ROC(symbol, sd_ind, ed, window=9)
        dfStock['psr'] = psr
        dfStock['bbv'] = bbv
        dfStock['roc'] = roc

        # Set parameters: N day return
        N = 1
        # Get trainX
        trainX = pd.concat((dfStock['psr'], dfStock['bbv'], dfStock['roc']), axis=1)
        trainX = trainX[:-N].values

        # Set YBuy，YSell
        Ndr = price[N:] / price[:-N].values - 1.0
        P = np.mean(price)
        e = 2.0 * self.commission / (2000.0 * P)
        YBuy = (e + self.impact * 2.0) / (1.0 - self.impact)
        YSell = (e - self.impact * 2.0) / (1.0 + self.impact)
        a = 1.2025
        b = 1.2020

        # Get trainY
        """
        trainY value instruction:
        2 = long
        1 = short
        0 = do nothing	   		  	  		  		  		    	 		 		   		 		  
        """
        trainY = []
        for i in range(numD - N):
            if Ndr[i] > a * YBuy:
                trainY.append(2)
            elif Ndr[i] < b * YSell:
                trainY.append(1)
            else:
                trainY.append(0)
        trainY = np.array(trainY)

        # In-sample training
        self.learner.add_evidence(trainX, trainY)
    """
    ------------------------------------------------------------------------------- 		  	   		  	  		  		  		    	 		 		   		 		  
                                    Out-of-sample Tests 		  	   		  	  		  		  		    	 		 		   		 		  
    -------------------------------------------------------------------------------
    Parameters:
    1. symbol (str): The stock symbol that you trained on on  		  	   		  	  		  		  		    	 		 		   		 		  
    2. sd (datetime): A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
    3. ed (datetime): A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
    4. sv (int): The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    Return: 		  	   		  	  		  		  		    	 		 		   		 		  
    df_trades (pandas.DataFrame): A DataFrame with values representing trades for each day. 
        Legal values are +1000.0 (a BUY of 1000 shares), -1000.0 (a SELL of 1000 shares), and 0.0 (do NOTHING).  		  	   		  	  		  		  		    	 		 		   		 		  
        Values of +2000 and -2000 for trades are also legal when switching from long to short or short to long 
        so long as net holdings are constrained to -1000, 0, and 1000.
    -------------------------------------------------------------------------------	  	   		  	  		  		  		    	 		 		   		 		  
    """
    def testPolicy(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):

        dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfStock[symbol].ffill().bfill()  # Filling missing date

        dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        dfStock = dfSPY.join(dfStock)  # Drop the days when market is closed
        numD = dfStock.shape[0]
        # Get indicators
        sd_ind = (sd - relativedelta(days=25)).strftime('%Y%m%d')
        sd_ind = dt.datetime.strptime(sd_ind, '%Y%m%d')
        psr = ind.PSRatio(symbol, sd_ind, ed, window=10)
        bbv = ind.BBV(symbol, sd_ind, ed, window=14)
        roc = ind.ROC(symbol, sd_ind, ed, window=9)
        dfStock['psr'] = psr
        dfStock['bbv'] = bbv
        dfStock['roc'] = roc
        # Get testX, testY (out-of-sample testing)
        testX = pd.concat((dfStock['psr'], dfStock['bbv'], dfStock['roc']), axis=1)
        testX = testX.values
        testY = self.learner.query(testX)
        # Get trades
        """
        testY value instruction:
        2 = long
        1 = short
        0 = do nothing	   		  	  		  		  		    	 		 		   		 		  
        """
        position = np.zeros(numD)
        trades = np.zeros(numD)
        for i in range(numD):
            if testY[i] == 2:
                position[i] = 1000
            elif testY[i] == 1:
                position[i] = -1000
            else:
                if i != 0:
                    position[i] = position[i - 1]

        trade = position[1:] - position[:-1]
        trades[0] = position[0]
        trades[1:] = trade[:]
        dfStock['Trade'] = trades
        df_trades = dfStock[['Trade']]
        return df_trades