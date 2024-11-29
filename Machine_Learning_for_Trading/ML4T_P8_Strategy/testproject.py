""""""
"""OMSCS2022FALL-P8: testproject 		  	   		  	  		  		  		    	 		 		   		 		  

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
import experiment1 as exp1
import experiment2 as exp2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

def author():
    return 'jhan446'
def gtid():
    return 903845311
'''
-------------------------------------------------------------------------------
                                    Test Code 
Wiki requires ManualStrategy.py, experiment1.py, experiment2.py generate charts,
so each scrip mentioned above has its own plot-charts function, and this scrip calls
these plot-charts functions from them to Generate all charts for P8 report
-------------------------------------------------------------------------------
'''
def test_code():
    np.random.seed(gtid())
    verbose = False
    ms.test_code(verbose)
    exp1.test_code(verbose)
    exp2.test_code(verbose)

if __name__ == "__main__":
    test_code()
