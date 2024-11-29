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
import math

def author():
    return 'jhan446'
'''
-------------------------------------------------------------------------------
        Get valid orders plus each stock's Adj close (result: dfOrders)
-------------------------------------------------------------------------------
Return:
dfOrder (pandas.DataFrame): valid orders when market is open + SPY Adj close + each stock's Adj close
stocks (str): stock symbols
num (int): number of rows in dfOrders >= valid trading days
numS (int): number of stocks among valid orders
-------------------------------------------------------------------------------
'''
def prepare_data(orders_file):
    # 1 Read orders & sort in time sequence
    dfOrders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    dfOrders = dfOrders.sort_index(ascending=True)
    # 2 Get date range in orders
    start_date = dfOrders.index[0]
    end_date = dfOrders.index[-1]
    # 3 Join SPY Adj Close and drop the day when the market is closed
    dfSPY = get_data(['SPY'], pd.date_range(start_date, end_date), addSPY=False, colname='Adj Close')
    dfOrders = dfSPY.join(dfOrders)
    num = dfOrders.shape[0]  # NOTE1: num >= valid trading days, this will be fixed in NOTE2
                             # if a day has multiple orders (N orders), the day will be counted N times
    # 4 Get stocks from the orders
    symbolsALL = dfOrders.loc[:, 'Symbol'].dropna()
    stocks = symbolsALL.unique()
    numS = stocks.shape[0]  # Number of stocks among valid orders
    # 5 Join stock Adj Close
    for stock in stocks:
        dfSymbol = get_data([stock], pd.date_range(start_date, end_date), addSPY=False, colname='Adj Close')
        dfSymbol = dfSymbol.ffill().bfill()  # Fill in missing stock prices
        dfOrders = dfOrders.join(dfSymbol)
    return dfOrders, stocks, num, numS
'''
-------------------------------------------------------------------------------
        Computes Daily Portfolio Values: cash + holding stock value(s)
-------------------------------------------------------------------------------
Parameters:
orders_file (str or file object): Path of the order file or the file object 		  	   		  	  		  		  		    	 		 		   		 		  
start_val (int ): The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  		  	   		  	  		  		  		    	 		 		   		 		  
commission (float): The fixed amount in dollars charged for each transaction (both entry and exit) 		  	   		  	  		  		  		    	 		 		   		 		  
impact (float): The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  	  		  		  		    	 		 		   		 		  

Return: 
portvals (pandas.DataFrame): A single-column dataframe, containing the value of the portfolio for each trading day 	
-------------------------------------------------------------------------------
'''
def compute_portvals(  		  	   		  	  		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",
    start_val=1000000,  		  	   		  	  		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		  	  		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		  	  		  		  		    	 		 		   		 		  
):
    dfOrders, stocks, num, numS = prepare_data(orders_file)

    # 1 Calculate daily share of stocks & daily cash (values change when buy/sell happen)
    orders = dfOrders.loc[:, 'Order']
    shares = dfOrders.loc[:, 'Shares']
    symbols = dfOrders.loc[:, 'Symbol']
    cash = np.empty(num)
    stockShare = np.zeros((numS, num))

    for id in range(num):
        if orders[id] == 'BUY':
            for i in range(numS):
                if symbols[id] == stocks[i]:
                    if id == 0:
                        stockShare[i, id] = 0 + shares[id]
                        cash[id] = start_val - shares[id] * dfOrders[stocks[i]][id] * (1.0 + impact) - commission
                    else:
                        stockShare[i, id] = stockShare[i, id - 1] + shares[id]
                        cash[id] = cash[id - 1] - shares[id] * dfOrders[stocks[i]][id] * (1.0 + impact) - commission
                else:
                    stockShare[i, id] = stockShare[i, id - 1]
        elif orders[id] == 'SELL':
            for i in range(numS):
                if symbols[id] == stocks[i]:
                    if id == 0:
                        stockShare[i, id] = 0 - shares[id]
                        cash[id] = start_val + shares[id] * dfOrders[stocks[i]][id] * (1.0 - impact) - commission
                    else:
                        stockShare[i, id] = stockShare[i, id - 1] - shares[id]
                        cash[id] = cash[id - 1] + shares[id] * dfOrders[stocks[i]][id] * (1.0 - impact) - commission
                else:
                    stockShare[i, id] = stockShare[i, id - 1]
        else:
            cash[id] = cash[id - 1]
            for i in range(numS):
                stockShare[i, id] = stockShare[i, id - 1]

    stockShare = stockShare.T
    for i in range(numS):
        dfOrders[str(stocks[i]) + 'S'] = stockShare[:, i]
    dfOrders['Cash'] = cash

    # 2 Drop the duplicated days when more than one order happen on a day, only keep the last entry of that day
    orders_date = dfOrders.index
    dfOrders['Dates'] = orders_date
    dfOrders = dfOrders.drop_duplicates(subset=['Dates'], keep='last')
    num = dfOrders.shape[0]     # NOTE2: num = valid trading days

    # 3 Calculate daily stock_val
    stock_val = np.zeros(num)
    for i in range(numS):
        stock_val = stock_val + dfOrders[stocks[i]] * dfOrders[str(stocks[i]) + 'S']

    # 4 Calculate daily portvals
    portvals = dfOrders['Cash'] + stock_val
    return portvals
'''
---------------------------------------------------------------------
                            Test Code
        during autograding this function will not be called
---------------------------------------------------------------------
'''
def test_code():

    path1 = './orders/'
    path2 = './orders/additional_orders/'
    ofs = [path1 + 'orders-01.csv', path1 + 'orders-02.csv', path1 + 'orders-03.csv', path1 + 'orders-04.csv',
           path1 + 'orders-05.csv',
           path1 + 'orders-06.csv', path1 + 'orders-07.csv', path1 + 'orders-08.csv', path1 + 'orders-09.csv',
           path1 + 'orders-10.csv',
           path1 + 'orders-11.csv', path1 + 'orders-12.csv', path2 + 'orders-short.csv', path2 + 'orders.csv',
           path2 + 'orders2.csv',
           path2 + 'orders-leverage-1.csv', path2 + 'orders-leverage-2.csv', path2 + 'orders-leverage-3.csv']

    for of in ofs:
        # 1. Process orders
        portvals = compute_portvals(orders_file = of)
        start_date = portvals.index[0]
        end_date = portvals.index[-1]
        if isinstance(portvals, pd.DataFrame):
            portvals = portvals[
                portvals.columns[0]]  # just get the first column
        else:
            "warning, code did not return a DataFrame"

        # 2. Get portfolio statistics
        fv = portvals[-1]
        portvals = portvals / portvals.iloc[0]
        cr = portvals[-1] / portvals[0] - 1
        dr = portvals[1:] / portvals[:-1].values - 1  # daily_return[0] does not exsit
        adr = dr.mean()
        sddr = dr.std()
        sr = math.sqrt(252) * (adr / sddr)

        # 3. Print results
#        print()
#        print('File name is:', {of})
#        print(f"Date Range: {start_date} to {end_date}")
#        print(f"Sharpe Ratio of Fund: {sr}")
#        print(f"Cumulative Return of Fund: {cr}")
#        print(f"Standard Deviation of Fund: {sddr}")
#        print(f"Average Daily Return of Fund: {adr}")
#        print(f"Final Portfolio Value: {fv}")
#        print( of.ljust(50), '%.6f' % sr, '   ', '%.4f' % fv)

if __name__ == "__main__":
    test_code()
