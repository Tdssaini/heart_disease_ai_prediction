from flask import Flask, request
from flask_restful import Api
import pickle
import pandas as pd
app = Flask(__name__)
api = Api(app)

@app.route("/predictHeartDisease", methods=["POST"])
def predictHeartDisease():
    data = request.get_json()
    print(data)
    age = data.get("age")
    sex = data.get("sex")
    chestPainType = data.get("chestPainType")
    restingBp = data.get("restingBp")
    cholesterol = data.get("cholesterol")
    fastingBs = data.get("fastingBs")
    restingECG = data.get("restingECG")
    maxHeartRate = data.get("maxHeartRate")
    exerciseAngina = data.get("exerciseAngina")
    oldPeak = data.get("oldPeak")
    stSlope = data.get("stSlope")
    predictionDataSet = "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope\n"
    predictionDataSet += str(age)+","+str(sex)+","+str(chestPainType)+","+str(restingBp)+","+str(cholesterol)+","+str(fastingBs)+","+str(restingECG)+","+str(maxHeartRate)+","+str(exerciseAngina)+","+str(oldPeak)+","+str(stSlope)
    X = pd.DataFrame([x.split(',') for x in predictionDataSet.split('\n')[1:]], columns=[x for x in predictionDataSet.split('\n')[0].split(',')])
    print(X)
    prediction = model.predict(X)
    print(prediction)
    return str(prediction.tolist()[0])

if __name__ == '__main__':
    with open('model_ab.pickle', 'rb') as f:
        model = pickle.load(f)
    app.run(debug=True)