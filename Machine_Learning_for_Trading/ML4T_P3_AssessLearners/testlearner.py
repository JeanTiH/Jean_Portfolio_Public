""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
"""
# Student Name: Juejing Han
# GT User ID: jhan446
# GT ID: 903845311

import math  		  	   		  	  		  		  		    	 		 		   		 		  
import sys
import numpy as np
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import matplotlib.pyplot as plt
import time

def author(self):
    return 'jhan446'  # GT username of the student
def gtid():
    return 903845311  # The GT ID of the student
'''
-------------------------------------------------
        Prepare Data to Train Learners
60% data for in_sample, 40% datat for out_sample
-------------------------------------------------
'''
def sort_data(data):
    # Split into data_x & data_y
    data_x = data[:, 0:-1]
    data_y = data[:, -1]
    # Randomly pick training and testing data
    num = data.shape[0]
    N = np.arange(num)
    train_index = np.random.choice(range(num), size=int(num * 0.6), replace=False)
    train_x = data_x[train_index]
    train_y = data_y[train_index]

    test_index = [i for i in N if i not in train_index]
    test_x = data_x[test_index]
    test_y = data_y[test_index]

    return train_x, train_y, test_x, test_y
'''
-------------------------------------------------
                    Experiments
-------------------------------------------------
'''
# Experiment 1 DTLearner
def exp1(data, max_leaf, runcount):

    for i in range(runcount):
        train_x, train_y, test_x, test_y = sort_data(data)
        in_rmse = []
        ou_rmse = []
        in_rmseS = np.zeros((runcount, max_leaf))
        ou_rmseS = np.zeros((runcount, max_leaf))

        for leaf in range(1, max_leaf + 1):
            # create a learner and train it
            learner = dtl.DTLearner(leaf_size = leaf, verbose = False)
            learner.add_evidence(train_x, train_y)  # train it

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
            in_rmse.append(rmse)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
            ou_rmse.append(rmse)

        in_rmseS[i] = in_rmse
        ou_rmseS[i] = ou_rmse

    in_mean = np.mean(in_rmseS, axis=0)
    ou_mean = np.mean(ou_rmseS, axis=0)

    # plot
    plt.xlim(0,max_leaf)
    plt.xticks(np.arange(0, max_leaf+5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE (10$^{-4}$)')
    plt.title('RMSE of DTLearner in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, in_mean * 10000, label='in sample')
    plt.plot(leaf, ou_mean * 10000, label='out of sample')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
    plt.savefig('Figure1.png')
    plt.close()

# Experiment 2 BagLearner
def exp2(data, max_leaf, bag_size, runcount, N):

    for i in range(runcount):
        train_x, train_y, test_x, test_y = sort_data(data)
        in_rmse = []
        ou_rmse = []
        in_rmseS = np.zeros((runcount, max_leaf))
        ou_rmseS = np.zeros((runcount, max_leaf))

        for leaf in range(1, max_leaf + 1):
            # create a learner and train it
            learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf}, bags=bag_size, boost=False,
                                verbose=False)
            learner.add_evidence(train_x, train_y)

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
            in_rmse.append(rmse)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
            ou_rmse.append(rmse)

        in_rmseS[i] = in_rmse
        ou_rmseS[i] = ou_rmse

    in_mean = np.mean(in_rmseS, axis=0)
    ou_mean = np.mean(ou_rmseS, axis=0)

    # plot
    plt.xlim(0,max_leaf)
    plt.xticks(np.arange(0, max_leaf+5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE (10$^{-4}$)')
    plt.title('RMSE of BagLearner (' + str(bag_size) + ' bags of DTLearner) in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, in_mean * 10000, label='in sample')
    plt.plot(leaf, ou_mean * 10000, label='out of sample')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
    plt.savefig('Figure' + str(N) + '.png')
    plt.close()

# Experiment 3 DTLearner vs. RTLearner (training time & MAE)
def exp3(data, max_leaf, runcount):

    for i in range(runcount):
        train_x, train_y, test_x, test_y = sort_data(data)
        time_dt = []
        time_rt = []
        dt_sum = np.zeros((runcount, max_leaf))
        rt_sum = np.zeros((runcount, max_leaf))

        in_dt = []
        ou_dt = []
        in_rt = []
        ou_rt = []
        in_dtS = np.zeros((runcount, max_leaf))
        ou_dtS = np.zeros((runcount, max_leaf))
        in_rtS = np.zeros((runcount, max_leaf))
        ou_rtS = np.zeros((runcount, max_leaf))

        for leaf in range(1, max_leaf + 1):
            # create a learner and train it
            learner = dtl.DTLearner(leaf_size = leaf, verbose = False)
            time_start = time.time()
            learner.add_evidence(train_x, train_y)
            time_end = time.time()
            time_dt.append(time_end - time_start)

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            mae = abs(train_y - pred_y).sum() / train_y.shape[0]
            in_dt.append(mae)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            mae = abs(test_y - pred_y).sum() / test_y.shape[0]
            ou_dt.append(mae)

            # create a learner and train it
            learner = rtl.RTLearner(leaf_size = leaf, verbose = False)
            time_start = time.time()
            learner.add_evidence(train_x, train_y)
            time_end = time.time()
            time_rt.append(time_end - time_start)

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            mae = abs(train_y - pred_y).sum() / train_y.shape[0]
            in_rt.append(mae)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            mae = abs(test_y - pred_y).sum() / test_y.shape[0]
            ou_rt.append(mae)

        dt_sum[i] = time_dt
        rt_sum[i] = time_rt

        in_dtS[i] = in_dt
        ou_dtS[i] = ou_dt
        in_rtS[i] = in_rt
        ou_rtS[i] = ou_rt

    dt_mean = np.mean(dt_sum, axis=0)
    rt_mean = np.mean(rt_sum, axis=0)

    in_dt_mean = np.mean(in_dtS, axis=0)
    ou_dt_mean = np.mean(ou_dtS, axis=0)
    in_rt_mean = np.mean(in_rtS, axis=0)
    ou_rt_mean = np.mean(ou_rtS, axis=0)

    # plot time
    plt.xlim(0,max_leaf)
    plt.xticks(np.arange(0, max_leaf+5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Time (10$^{-3}$s)')
    plt.title('DTLearner vs. RTLearner on Training Time in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, dt_mean * 1000, label='DTLearner')
    plt.plot(leaf, rt_mean * 1000, label='RTLearner')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
    plt.savefig('Figure4.png')
    plt.close()

    # plot MAE
    plt.xlim(0, max_leaf)
    plt.xticks(np.arange(0, max_leaf + 5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE (10$^{-4}$)')
    plt.title('DTLearner vs. RTLearner on MAE in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, in_dt_mean * 10000, 'b', label='in sample(DT)')
    plt.plot(leaf, ou_dt_mean * 10000, 'g', linestyle = 'dashed', label='out of sample(DT)')
    plt.plot(leaf, in_rt_mean * 10000, 'r',  label='in sample(RT)')
    plt.plot(leaf, ou_rt_mean * 10000, 'y', linestyle = 'dashed', label='out of sample(RT)')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
    plt.savefig('Figure5.png')
    plt.close()

# Experiment 4 DTLearner vs. Bagged RTLearner
def exp4(data, max_leaf, bag_size, runcount):
    for i in range(runcount):
        train_x, train_y, test_x, test_y = sort_data(data)
        time_dt = []
        time_rt = []
        dt_sum = np.zeros((runcount, max_leaf))
        rt_sum = np.zeros((runcount, max_leaf))

        in_dt = []
        ou_dt = []
        in_rt = []
        ou_rt = []
        in_dtS = np.zeros((runcount, max_leaf))
        ou_dtS = np.zeros((runcount, max_leaf))
        in_rtS = np.zeros((runcount, max_leaf))
        ou_rtS = np.zeros((runcount, max_leaf))

        for leaf in range(1, max_leaf + 1):
            # create a learner and train it
            learner = dtl.DTLearner(leaf_size = leaf, verbose = False)
            time_start = time.time()
            learner.add_evidence(train_x, train_y)  # train it
            time_end = time.time()
            time_dt.append(time_end - time_start)

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            mae = abs(train_y - pred_y).sum() / train_y.shape[0]
            in_dt.append(mae)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            mae = abs(test_y - pred_y).sum() / test_y.shape[0]
            ou_dt.append(mae)

            # create a learner and train it
            learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": leaf}, bags=bag_size, boost=False,
                                verbose=False)
            time_start = time.time()
            learner.add_evidence(train_x, train_y)  # train it
            time_end = time.time()
            time_rt.append(time_end - time_start)

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            mae = abs(train_y - pred_y).sum() / train_y.shape[0]
            in_rt.append(mae)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            mae = abs(test_y - pred_y).sum() / test_y.shape[0]
            ou_rt.append(mae)

        dt_sum[i] = time_dt
        rt_sum[i] = time_rt

        in_dtS[i] = in_dt
        ou_dtS[i] = ou_dt
        in_rtS[i] = in_rt
        ou_rtS[i] = ou_rt

    dt_mean = np.mean(dt_sum, axis=0)
    rt_mean = np.mean(rt_sum, axis=0)

    in_dt_mean = np.mean(in_dtS, axis=0)
    ou_dt_mean = np.mean(ou_dtS, axis=0)
    in_rt_mean = np.mean(in_rtS, axis=0)
    ou_rt_mean = np.mean(ou_rtS, axis=0)

    # plot time
    plt.xlim(0,max_leaf)
    plt.xticks(np.arange(0, max_leaf+5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Time (10$^{-3}$s)')
    plt.title('DTLearner vs. ' + str(bag_size) + '-Bagged RTLearner on Training Time in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, dt_mean * 1000, label='DTLearner')
    plt.plot(leaf, rt_mean * 1000, label=str(bag_size) + '-Bagged RTLearner')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
#    plt.savefig('Figure6-time.png')
    plt.close()

    # plot MAE
    plt.xlim(0, max_leaf)
    plt.xticks(np.arange(0, max_leaf + 5, 5))
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE (10$^{-4}$)')
    plt.title('DTLearner vs. ' + str(bag_size) + '-Bagged RTLearner on MAE in ' + str(runcount) + ' trials')

    leaf = range(1, max_leaf + 1)
    plt.plot(leaf, in_dt_mean * 10000, 'b', label='in sample(DT)')
    plt.plot(leaf, ou_dt_mean * 10000, 'g', linestyle = 'dashed', label='out of sample(DT)')
    plt.plot(leaf, in_rt_mean * 10000, 'r',  label='in sample(' + str(bag_size) + '-Bagged RT)')
    plt.plot(leaf, ou_rt_mean * 10000, 'y', linestyle = 'dashed', label='out of sample(' + str(bag_size) + '-Bagged RT)')
    plt.grid(color='gainsboro', linestyle='dashed')
    plt.legend()
    plt.savefig('Figure6.png')
    plt.close()
'''
-------------------------------------------------
        Read in Data and Run Experiments
-------------------------------------------------
'''
if __name__ == "__main__":
#    startall = time.time()
    np.random.seed(gtid())
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    inf = open(sys.argv[1])
    if sys.argv[1] == 'Data/Istanbul.csv':
        data = np.array(
            [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]   # Remove title and time
        )
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
    # Run each expriment 10 times
    exp1(data, 100, 10)         # Input data, max leaf size, runcount
    exp2(data, 100, 2, 10, 2)      # Input data, max leaf size, bag size, runcount, Fig number
    exp2(data, 100, 20, 10, 3)     # Input data, max leaf size, bag size, runcount, Fig number
    exp3(data, 100, 10)         # Input data, max leaf size, runcount
    exp4(data, 100, 8, 10)     # Input data, max leaf size, bag size, runcount
#    endall = time.time()
#    print(endall - startall)