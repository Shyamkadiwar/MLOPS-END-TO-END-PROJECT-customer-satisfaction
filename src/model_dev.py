import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class model(ABC):
    """
    abstract class for all model
    """
    
    @abstractmethod
    def train(self,X_train,y_train):
        pass


class LinearRegressionModel(model):
    def train(self,X_train,y_train,**kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("model train completed")
            return reg
        except Exception as e:
            logging.error("Error in model training: {}".format(e))
            raise e