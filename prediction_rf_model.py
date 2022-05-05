import pandas as pd
import pickle 
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline

data = pd.read_csv('data.csv')
print(data)

print(data.isnull().sum())
print(data.head())
print(data.shape)
x = data.iloc[:,0:11]
y=data.HeartDisease
print(x)
print(y)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(data.head)
print(data.ST_Slope.unique())

nomi_col  = [1,2,8,10]
ordi_col = [6]
trans = make_column_transformer((OneHotEncoder(sparse = False),nomi_col),
                               (OrdinalEncoder(),ordi_col),
                               remainder = 'passthrough')
set_config(display = 'diagram')
print(trans)

model_R =  RandomForestClassifier(n_estimators = 400,min_samples_split = 5,min_samples_leaf = 3)
pipe_R = make_pipeline(trans,model_R)
print(pipe_R)
pipe_R.fit(x_train,y_train)
pred_R = pipe_R.predict(x_test)
print('Accuracy for RandomForestClassifier')
print(accuracy_score(pred_R,y_test)*100)

print(model_R.feature_importances_)
print(x.columns)

prediction_data = "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope\n"
prediction_data += "40,M,ATA,140,289,0,Normal,172,N,0,Up"
new = pd.DataFrame([x.split(',') for x in prediction_data.split('\n')[1:]], columns=[x for x in prediction_data.split('\n')[0].split(',')])
print(new)
print(type(new))
print(pipe_R.predict(new))

with open('model_rf.pickle','wb') as f:
    pickle.dump(pipe_R,f)