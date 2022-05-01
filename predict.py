from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import pandas as pd

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')

class IrisClassifier(Resource):
    def post(self):
        X = pd.read_csv('prediction.csv')
        print(X)
        prediction = model.predict(X)
        print(prediction)
        return prediction.tolist()

api.add_resource(IrisClassifier, '/iris')

if __name__ == '__main__':
    # Load model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    app.run(debug=True)