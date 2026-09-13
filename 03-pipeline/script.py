import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance"
    ]

    df = pd.read_parquet(url, columns=columns)
    df=df.head(1000)  # For testing purposes, limit to first 1000 rows
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    df[categorical] = df[categorical].astype(str)

    return df

def create_X(df,dv=None):
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train,X_val, y_train, y_val, dv):
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "your_name")
        mlflow.set_tag("model", "xgboost")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=30,
            evals=[(dval, "validation")],
            early_stopping_rounds=20
        )
        y_pred = booster.predict(dtrain)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        return run.info.run_id

def run_pipeline(year, month):
    df_train = read_dataframe(year, month)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(next_year, next_month)
    X_train, dv = create_X(df_train)
    y_train = df_train.duration.values
    X_val, _ = create_X(df_val, dv)
    y_val = df_val.duration.values

    run_id = train_model(X_train, X_val, y_train, y_val, dv)
    print(f"Model trained and logged with run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    run_id = run_pipeline(year=args.year, month=args.month)
    with open("run_id.txt", "w") as f:
        f.write(run_id)