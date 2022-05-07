## AI Prediction Model to predict if patient has Heart Disease 

In this project, I have developed AI prediction model using 3 different classification techniques -

1. AdaBoost Classification
2. K-Nearest Neighbor(KNN)
3. Random Forest

For this type of use case, AdaBoost perfomed best with highest accuracy of approx 93%. 

In this project, I have all these 3 model trained using Python and also have the pickle file to start prediction. For the prediction, I have created an API using Flask which reads the pickle file and then predicts if patient has Heart Disease. 

To train the model, you can run below command -

python {{model_file_name}}.py  //example python prediction_knn_model.py 


To do the prediction, you can run below command which runs the API server -

python predict_api.py or heroku local (if you have heroku cli setup locally)
