
# InsaneLearner

# Student Name: Juejing Han
# GT User ID: jhan446
# GT ID: 903845311

import numpy as np
import BagLearner as bl
import LinRegLearner as lrl	   	  			  	 		  		  		    	 		 		   		 		  	 		  		  		    	 		 		   		 		  
class InsaneLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False):
        self.learners = []
        for i in range(20):
            self.learners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False))
    def author(self):
        return 'jhan446'
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self,points):
        query_result = []
        for learner in self.learners:
            query_result.append(learner.query(points))
        return np.mean(np.asarray(query_result), axis = 0)