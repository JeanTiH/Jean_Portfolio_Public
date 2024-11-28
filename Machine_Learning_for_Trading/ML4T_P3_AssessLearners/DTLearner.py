
# Regression Decision Tree Leaner - DTLearner

# Student Name: Juejing Han
# GT User ID: jhan446
# GT ID: 903845311

import numpy as np
'''
---------------------------------------------------------------------
                            DTLearner
---------------------------------------------------------------------
'''
class DTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size, verbose = False):     # verbose = False means no print-out information
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'jhan446'       # GT username of the student
    '''
    Add training data (data_x,data_y) to the decision tree
    '''
    def add_evidence(self,data_x, data_y):
        self.tree_result = self.decisionTree(data_x, data_y)
    '''
    Estimate a set of test points based on the decision tree
    '''
    def query(self, points):
        query_result = np.empty(points.shape[0])    # Creat an empty array share the same size of data_y from points
        iq = -1
        for point in points:
            # Relative references
            node = 0
            iq = iq + 1
            while self.tree_result[node][0] != -1:  # Not a leaf
                split_index = int(self.tree_result[node][0])
                split_val = point[split_index]
                if split_val <= self.tree_result[node][1]:          # Go to leftTree
                    node = node + int(self.tree_result[node][2])
                else:                                               # Go to rightTree
                    node = node + int(self.tree_result[node][3])
            query_result[iq] = self.tree_result[node][1]
        return query_result
    '''
    ---------------------------------------------
                    Build The Decision Tree
    return a numpy.array[feature, split_val, left_tree, right_tree]
    
    feature: leaf_index = -1 
    highest abs(correlation) is used to determine the split_index and split_val
    left_index/right_index is the Boolean values labeling data_x[slpit_index] belongs to left/right tree
    left_tree is the entries of data_x belong to left tree
    right_tree is the entries of data_x belong to right tree
    ---------------------------------------------
    '''
    def decisionTree(self, data_x, data_y):
        leaf_index = -1

        # 1. Generate leaf when no need to sort data
        if data_x.shape[0] <= self.leaf_size or data_x.ndim == 1:
            return np.asarray([leaf_index, np.median(data_y), np.nan, np.nan])
        # All data_y share the close y value, generate the leaf
        if np.all(np.isclose(data_y, data_y[0])):
            return np.asarray([leaf_index, np.mean(data_y), np.nan, np.nan])
        # All data_x share the close x value, generate the leaf
        if np.all(np.isclose(data_x, data_x[0])):
            return np.asarray([leaf_index, np.median(data_y), np.nan, np.nan])

        # 2. Locate the best feature and split_val based on the highest absolute value of x and y
        num = data_x.T.shape[0]
        np.seterr(invalid='ignore')     # When std=0.0 correlation='nan', ignore warnings
        correlation = np.corrcoef(data_x.T, data_y)[:,num]
        correlation[np.isnan(correlation)] = 0.0    # eliminate correlation='nan', tramsfer 'nan' to 0.0
        split_index = abs(correlation[0:num]).argmax(axis=0)
        split_val = np.median(data_x[:, split_index])
        # Prevent all data goes to one side (e.g., data_x[split_index]=[1.9,2,2], use median would be a trouble)
        if split_val == np.max(data_x[:,split_index]):
            split_val = np.mean(data_x[:, split_index])

        # 3. Split into left and righ tree
        left_index = data_x[:, split_index] <= split_val    # True means data_x[slpit_index]  belongs to left tree
        #  All data goes to left/right tree, generate the leaf
        if np.all(left_index == True) or np.all(left_index == False):
            return np.asarray([leaf_index, np.mean(data_y), np.nan, np.nan])

        right_index = data_x[:, split_index] > split_val    # True means data_x[slpit_index]  belongs to right tree
        '''
        The following code refers to the slide from "CS 7646: How to build a decision tree from data"
        '''
        # 4. Recursive process
        leftTree = self.decisionTree(data_x[left_index], data_y[left_index])
        rightTree = self.decisionTree(data_x[right_index], data_y[right_index])

        if leftTree.ndim == 1:
            root = np.asarray([split_index, split_val, 1, leftTree.ndim + 1])
        else:
            root = np.asarray([split_index, split_val, 1, leftTree.shape[0] + 1])
        return np.vstack((root, leftTree, rightTree))
