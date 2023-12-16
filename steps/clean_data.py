import logging
from typing import Tuple

import pandas as pd
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStraregy
from typing_extensions import Annotated

# from zenml.steps import Output, step
from zenml import step


@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        preprocess_strategy = DataPreProcessStraregy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handel_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handel_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e



# import logging
# import pandas as pd
# from zenml import step
# from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStraregy
# from typing_extensions import Annotated
# from typing import Tuple

# @step
# def clean_df(df: pd.DataFrame) -> tuple[
#     Annotated[pd.DataFrame,"X_train"],
#     Annotated[pd.DataFrame,"X_test"],
#     Annotated[pd.Series,"y_train"],
#     Annotated[pd.Series,"y_test"]
#      ]:
#     try:
#         process_strategy = DataPreProcessStraregy()
#         data_cleaning = DataCleaning(df,process_strategy)
#         processed_data = data_cleaning.handel_data()

#         divide_strategy = DataDivideStrategy()
#         data_cleaning = DataCleaning(df,process_strategy)
#         X_train,X_test,y_train,y_test = data_cleaning.handel_data()
#         logging.info("data cleanig completed")
#         return X_train,X_test,y_train,y_test
    
#     except Exception as e:
#         logging.error("Error i cleaning data: {}".format(e))
#         raise e