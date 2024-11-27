""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import scipy.optimize as spo
import math
"""
---------------------------------------------------------------------
                1. Portfolio Optimization (minimization)
---------------------------------------------------------------------
import: start_date,end-date
        symbols(i.e., stocks)
        gen_plot = True for local resport fig only
        
output: allocs_result   optimized allocation index of each stock
        cr              Cumulative Return
        adr             Average Daily Return
        sddr            Standard Deviation of Daily Return
        sr              Sharpe Ratio
---------------------------------------------------------------------
"""
# This is the function that will be tested by the autograder
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):
  		  	   		  	  		  		  		    	 		 		   		 		  
    # 1. Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)  		  	   		  	  		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
    prices = prices_all[syms]           # only portfolio symbols
    prices_SPY = prices_all['SPY']      # only SPY, for comparison later
    prices = prices.ffill().bfill()     # fill missing data
  		  	   		  	  		  		  		    	 		 		   		 		  
    # 2. Find the allocations for the optimal portfolio
    num = len(syms)                             # Numbers of stocks in the portfolio
    allocs_guess = np.full((num), 1.0 / num)    # Initial guess of all allocs (evenly distributed)
    fun = lambda allocs: - statistics(allocs, prices)[4]              # Find minimum -(Sharpe Ratio), i.e., maximum sr
    cons = [{'type': 'eq', 'fun': lambda allocs: 1.0 - sum(allocs)}]  # Sum(allocs) of all assets should be 1.0
    bous = [(0, 1)] * num                                             # Each alloc should be [0,1]
    result = spo.minimize(fun, allocs_guess, method='SLSQP', constraints=cons, bounds=bous)
    allocs_result = result.x

    # 3. Get daily portfolio value (i.e., normalized daily port_val)
    port_val, cr, adr, sddr, sr = statistics(allocs_result, prices)
    # Get normalized SPY for comparison
    normed_SPY = prices_SPY / prices_SPY[0]
  		  	   		  	  		  		  		    	 		 		   		 		  
    # 4.Compare daily portfolio value with SPY using a normalized plot
    """
    This is for generating local report fig ONLY
    """
    if gen_plot:
        df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot(color='br',grid=True)

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Daily Portfolio Value & SPY')
        plt.grid(color='gainsboro',linestyle='dashed')
        plt.savefig('Figure1.png')
        plt.close()

    return allocs_result, cr, adr, sddr, sr
"""
---------------------------------------------------------------------
            2. Calculate Statistics for Portfolio (Normalized)
---------------------------------------------------------------------
cr        Cumulative Return
adr       Average Daily Return
sddr      Standard Deviation of Daily Return
sr        Sharpe Ratio
port_val  also normalized daily value of portfolio
---------------------------------------------------------------------
"""
def statistics(allocs, prices):
    # 1. Get port_val
    normed_prices = prices / prices.iloc[0]
    allocs_prices = normed_prices * allocs
    port_val = allocs_prices.sum(axis=1)

    # 2. Get cr, daily return
    cr = port_val[-1] / port_val[0] - 1

    daily_return = port_val[1:] / port_val[:-1].values - 1  # daily_return[0] does not exsit

    # 3. Get adr, sddr, sr
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = math.sqrt(252) * (adr / sddr)

    return port_val, cr, adr, sddr, sr
  		  	   		  	  		  		  		    	 		 		   		 		  
def test_code():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader
    For local report fig ONLY  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
    # Assess the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )
  		  	   		  	  		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    test_code()  		  	   		  	  		  		  		    	 		 		   		 		  
