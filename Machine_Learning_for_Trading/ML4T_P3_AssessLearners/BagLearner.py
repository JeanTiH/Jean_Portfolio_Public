
# Bootstrap Aggregation Leaner - BagLearner

# Student Name: Juejing Han
# GT User ID: jhan446
# GT ID: 903845311

import numpy as np
'''
---------------------------------------------------------------------
                            BagLearner
---------------------------------------------------------------------
'''
class BagLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  	# kwargs will be set as leaf size
    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        # Generae ensemble learners (number of learners = bags)
        '''
        This clip of code refers to the instruction of P3
        '''
        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**self.kwargs))
 
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'jhan446'
  		   	  			  	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            # Generate randomly picked training data for each learner/bag
            num = data_x.shape[0]
            train_index = np.random.choice(range(num), size = num, replace = True)
            train_x = data_x[train_index]
            train_y = data_y[train_index]
            # Train each learner/bag
            learner.add_evidence(train_x, train_y)

    def query(self, points):
        # Get ensemble predict_y
        query_result = []
        for learner in self.learners:
            query_result.append(learner.query(points))
        # Return ensemble mean of predict_y
        query_ensemble_mean = np.mean(np.asarray(query_result),axis = 0)
        return query_ensemble_mean