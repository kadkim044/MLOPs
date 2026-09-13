from prefect import task,flow

@task
def get_data():
    print("Loading data...")
    return [1,2,3,4,5]

@task
def calculate_average(data):
    average=sum(data)/len(data)
    print(f"Average:{average}")
    return average

@flow
def data_flow():
    data=get_data()
    average=calculate_average(data)

if __name__=="__main__":
    data_flow()