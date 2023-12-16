from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "LinearRegression"
    fine_tuning: bool = False

# from zenml.steps import base_parameters

# class ModelNameConfig(base_parameters):
#     model_name = "LinearRegression"
#     # model_name: str = "lightgbm"
#     fine_tuning: bool = False