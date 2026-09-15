import pickle
from flask import Flask,request,jsonify
with open("lin_reg.bin","rb") as f_in:
    (dv,model)=pickle.load(f_in)


def predict(features):
    X=dv.transform(features)
    preds=model.predict(X)
    return preds


app=Flask("ride-duration-prediction")

@app.route("/predict",methods=["POST"])
def predict_endpoints():
    ride=request.get_json()
    pred=predict(ride)
    result={
        "prediction":float(pred[0])
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=9696)