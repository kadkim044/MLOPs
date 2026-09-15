import mlflow
from flask import Flask,request,jsonify
mlflow.set_tracking_uri("http://host.docker.internal:5000")
model=mlflow.sklearn.load_model("models:/california-housing-model/1")
app=Flask("california-housing-prediction")
@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([[
        data["MedInc"],
        data["HouseAge"],
        data["AveRooms"],
        data["AveBedrms"],
        data["Population"],
        data["AveOccup"],
        data["Latitude"],
        data["Longitude"]
    ]])
    print("Prediction:", prediction)
    return jsonify({"prediction": prediction[0]})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)