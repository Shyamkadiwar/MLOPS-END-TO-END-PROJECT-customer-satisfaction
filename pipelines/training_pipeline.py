from zenml import pipeline
from steps.ingest_data import ingestdf
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingestdf()
    X_train,X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    r2,rmse = evaluate_model(model,X_test,y_test)