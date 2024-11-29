""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
import numpy as np
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
from util import get_data, plot_data
'''
-------------------------------------------------------------------------------
'''
def author():
    return 'jhan446'
'''
-------------------------------------------------------------------------------
            Computes Daily Portfolio Values: cash + holding stock value
-------------------------------------------------------------------------------
Parameters:
df_trades (pandas.DataFrame): A single-column dataframe, containing the trades of a stock (JPM) on each trading day 	  	   		  	  		  		  		    	 		 		   		 		  
sv (int ): The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  		  	   		  	  		  		  		    	 		 		   		 		  
commission (float): The fixed amount in dollars charged for each transaction (both entry and exit) 		  	   		  	  		  		  		    	 		 		   		 		  
impact (float): The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  	  		  		  		    	 		 		   		 		  

Return: 
portvals (pandas.DataFrame): A single-column dataframe, containing the value of the portfolio for each trading day 	
-------------------------------------------------------------------------------
'''
def compute_portvals(df_trades, symbol, sv=1000000, commission=9.95, impact=0.005):

    sd = df_trades.index[0]
    ed = df_trades.index[-1]

    # Get stock price and fill missing data
    dfStock = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfStock[symbol].ffill().bfill()   # Filling missing date
    # Get SPY and eliminate non-trading days
    dfSPY = get_data(['SPY'], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    dfStock = dfSPY.join(dfStock)   # Drop the days when market is closed
    dfOrders = dfStock.join(df_trades)
    num = dfOrders.shape[0]

    # Calculate daily cash (values change when buy/sell happen, i.e., order != 0) & stock_val
    orders = dfOrders['Trade']
    price = dfOrders[symbol]
    cash = np.empty(num)
    stock_val = np.empty(num)
    stockShare = np.zeros((num))

    i = -1
    for order in orders:
        i = i + 1
        if i == 0:
            stockShare[i] = order
            if order == 0:
                cash[i] = sv - order * price[i] * (1.0 + impact)
            else:
                cash[i] = sv - order * price[i] * (1.0 + impact) - commission
        else:
            stockShare[i] = stockShare[i-1] + order
            if order == 0:
                cash[i] = cash[i-1]- order * price[i] * (1.0 + impact)
            else:
                cash[i] = cash[i - 1] - order * price[i] * (1.0 + impact) - commission
        stock_val[i] = stockShare[i] * price[i]

    # Calculate daily portvals
    dfOrders['Cash'] = cash
    portvals = dfOrders['Cash'] + stock_val
    return portvals