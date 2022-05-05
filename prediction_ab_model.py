import pandas as pd
import pickle 
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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

model_ab = AdaBoostClassifier()
pipe_ab = make_pipeline(trans,model_ab) 
pipe_ab
pipe_ab.fit(x_test,y_test)
pred_ab = pipe_ab.predict(x_test)
pred_ab
print('Accuracy for AdaBoostClassifier')
print(accuracy_score(pred_ab,y_test)*100)


params_ad = {'n_estimators':[20,25,15,34,40,55],
          'learning_rate':[0.01,0.03,0.06,1,1.54]}
pipe_ab1 = make_pipeline(trans,GridSearchCV(model_ab,params_ad,cv = 3,n_jobs = -1,verbose = 3))
print(pipe_ab1)
pipe_ab1.fit(x_train,y_train)
pred_ab1 = pipe_ab1.predict(x_test)
print(pred_ab1)
print('Accuracy for AdaBoostClassifier GridSearchCV 1')
print(accuracy_score(pred_ab1,y_test)*100)


params_ad1 = {'n_estimators':[20],
          'learning_rate':[1.54]}
pipe_ab2 = make_pipeline(trans,GridSearchCV(model_ab,params_ad1,cv = 3,n_jobs = -1,verbose = 3))
print(pipe_ab2)
pipe_ab2.fit(x_train,y_train)
pred_ab2 = pipe_ab2.predict(x_test)
print('Accuracy for AdaBoostClassifier GridSearchCV 2')
print(accuracy_score(pred_ab2,y_test)*100)

print(model_ab.feature_importances_)
print(x.columns)

prediction_data = "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope\n"
prediction_data += "40,M,ATA,140,289,0,Normal,172,N,0,Up"
new = pd.DataFrame([x.split(',') for x in prediction_data.split('\n')[1:]], columns=[x for x in prediction_data.split('\n')[0].split(',')])
print(new)
print(type(new))
print(pipe_ab.predict(new))

with open('model_ab.pickle','wb') as f:
    pickle.dump(pipe_ab,f)
    