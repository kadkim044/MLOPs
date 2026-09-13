from prefect import task,flow
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
@task(retries=3, retry_delay_seconds=5)
def load_data(n_rows):
    np.random.seed(42)
    df=pd.DataFrame({
        "PULocationID": np.random.randint(1, 6, n_rows),
        "DOLocationID": np.random.randint(1, 6, n_rows),
        "trip_distance": np.random.uniform(1, 20, n_rows),
        "duration": np.random.uniform(5, 60, n_rows)
    })
    print(f"Loaded {len(df)} records")
    return df
@task
def prepare_data(df):
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    df[categorical] = df[categorical].astype(str)

    dicts = df[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer()
    X = dv.fit_transform(dicts)

    y = df["duration"].values

    return X, y, dv
@task
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    return model
@task
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)

    print(f"RMSE: {rmse}")

    return rmse
@task
def log_model(model,dv,rmse):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("small-taxi-prefect")
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            model,
            "model"
        )

    print("Model logged to MLflow")

@flow(name="training-flow",log_prints=True)
def training_pipeline(n_rows=5000):
    df = load_data(n_rows)

    X, y, dv = prepare_data(df)

    model = train_model(X, y)

    rmse = evaluate_model(model, X, y)
    log_model(model, dv, rmse)
    print(f"Final RMSE: {rmse}")

if __name__=="__main__":
    training_pipeline()