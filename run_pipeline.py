from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
import mlflow

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    train_pipeline(data_path="D:\mlops project\data\olist_customers_dataset.csv")

#mlflow ui --backend-store-uri "file:C:\Users\shyam\AppData\Roaming\zenml\local_stores\6ef64c2d-5da2-40e3-b24e-dc662027f639\mlruns"