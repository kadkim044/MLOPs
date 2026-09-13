import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from prefect import task,flow

@task
def load_data():
    data=fetch_california_housing()
    X=pd.DataFrame(data=data["data"],columns=data["feature_names"])
    y=pd.Series(data=data["target"],name="MedHouseVal")
    print(f"Loaded {len(X)} records.")
    return X,y
@task
def prepare_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    print(f"Data Prepared.")
    return X_train,X_test,y_train,y_test
@task
def train_model(X_train,y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    print(f"Model trained.")
    return model
@task
def evaluate_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    rmse=root_mean_squared_error(y_test,y_pred)
    print(f"RMSE:{rmse:.4f}")
    return rmse
@task
def log_model(model,rmse):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("california-housing-price-prediction")
    with mlflow.start_run():
        mlflow.log_param("model","LinearRegression")
        mlflow.log_metric("rmse",rmse)
        mlflow.sklearn.log_model(model,"model")
    print("Model logged to MLflow")
@flow(name="training-flow",log_prints=True)
def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=prepare_data(X,y)
    model=train_model(X_train,y_train)
    rmse=evaluate_model(model,X_test,y_test)
    log_model(model,rmse)
if __name__=="__main__":
    main()