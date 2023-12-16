import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    abstract class for evaluating model
    """
    
    @abstractmethod
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        pass

class MSE(Evaluation):
    #model which use mean squared error
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("mse: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
        #model which use r2_score
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating r2_score")
            r2 = r2_score(y_true,y_pred)
            logging.info("r2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in r2: {}".format(e))
            raise e
        
class RMSE(Evaluation):
        #model which use mean squared error
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating RMSE")
            rmse = mean_squared_error(y_true,y_pred,squared=False)
            logging.info("rmse: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in RMSE: {}".format(e))
            raise e 