import logging
from abc import ABC,abstractmethod
import pandas as pd
import numpy as np
from typing import Union

from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    class fr handeling the data
    """
    @abstractmethod
    def handel_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass


class DataPreProcessStraregy(DataStrategy):
    # for preprocessesing data
    def handel_data(self,data:pd.DataFrame) -> pd.DataFrame:
        try:
            data=data.drop(
                [
                 "order_approved_at", 
                 "order_delivered_carrier_date",
                 "order_delivered_customer_date",
                 "order_estimated_delivery_date",
                 "order_purchase_timestamp"
                ],axis=1
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna (data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna (data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna (data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            
            data = data.select_dtypes(include=[np.number]) # we are only selecting int data for model training
            col_to_drop =["customer_zip_code_prefix","order_item_id"]
            return data
        except Exception as e:
            logging.error("error in processeing data: {}".format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):
    """
    class for dividing data into train and test  
    """
    def handel_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X=data.drop(["review_score"],axis=1)
            y=data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("error in diving data: {}".format(e))
            raise e
        

class DataCleaning:
    """
    class for cleaning data which process and divide data into train test
    """
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data = data
        self.strategy = strategy

    def handel_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handel_data(self.data)
        except Exception as e:
            logging.error("error in handelling data: {}".format(e))
            raise e