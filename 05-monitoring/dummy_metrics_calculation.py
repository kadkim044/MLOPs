import datetime,time,random,logging,uuid,pytz,io,psycopg,joblib
import pandas as pd
from evidently.presets import DataDriftPreset,DataSummaryPreset
from evidently import Report
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT=10
rand=random.Random()
create_table_statement="""
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

numeric_features=["passenger_count","trip_distance","fare_amount","total_amount"]
categorical_features=["PULocationID","DOLocationID"]

reference_data=pd.read_parquet("data/reference.parquet")
with open("models/lin_reg.bin","rb") as f_in:
    model=joblib.load(f_in)

raw_data=pd.read_parquet("data/green_tripdata_2022-02.parquet")
begin=datetime.datetime(2022,2,1,0,0)    

report=Report(
    metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ]
)

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example",autocommit=True) as conn:
        res=conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall())==0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

def calculate_metrics_postgresql(curr,i):
    current_data=raw_data[(raw_data.lpep_pickup_datetime<=(begin+datetime.timedelta(i))) &
                          (raw_data.lpep_pickup_datetime<=(begin+datetime.timedelta(i+1)))]
    current_data[numeric_features] = current_data[numeric_features].fillna(0)
    current_data[categorical_features] = current_data[categorical_features].fillna(0)
    current_data["preds"]=model.predict(current_data[numeric_features+categorical_features])
    snapshot=report.run(
    current_data=current_data,
    reference_data=reference_data
    )
    snapshot.save_html("snap.html")
    result = snapshot.dict()

    prediction_drift = next(
        m["value"]
        for m in result["metrics"]
        if "ValueDrift(column=preds" in m["metric_name"]
    )

    num_drifted_columns = next(
        m["value"]["count"]
        for m in result["metrics"]
        if "DriftedColumnsCount" in m["metric_name"]
    )

    share_missing_values = next(
        m["value"]["share"]
        for m in result["metrics"]
        if m["metric_name"] == "DatasetMissingValueCount()"
    )

    curr.execute(
        "insert into dummy_metrics(timestamp,prediction_drift,num_drifted_columns,share_missing_values) values (%s,%s,%s,%s)",
        (datetime.datetime.now(pytz.timezone("Europe/Paris")),prediction_drift,num_drifted_columns,share_missing_values)
    )

def main():
    prep_db()
    last_send=datetime.datetime.now()-datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example",autocommit=True) as conn:
        for i in range(0,27):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr,i)
            new_send=datetime.datetime.now()
            seconds_elapsed=(new_send-last_send).total_seconds()
            if seconds_elapsed<SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT-seconds_elapsed)
            while last_send<new_send:
                last_send=last_send+datetime.timedelta(seconds=10)
            logging.info("data sent")

if __name__=="__main__":
    main(   )