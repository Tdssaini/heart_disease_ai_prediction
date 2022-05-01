import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle 
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from xgboost import plot_importance

data = pd.read_csv('heart raw_Data.csv')
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
KNN = KNeighborsClassifier(5)
pipe_KNN = make_pipeline(trans,KNN)
print(KNN)
print(pipe_KNN)
pipe_KNN.fit(x_train,y_train)
pred_KNN = pipe_KNN.predict(x_test)
print(pred_KNN)
print(accuracy_score(pred_KNN,y_test)*100)

prediction_data = pd.read_csv('prediction.csv')
print(prediction_data)
print(type(prediction_data))
new = pd.DataFrame(prediction_data)
print(pipe_KNN.predict(new))


y.value_counts()
sns.countplot(y)
under = RandomUnderSampler()
u_x,u_y = under.fit_resample(x,y)
u_y.value_counts()
over = RandomOverSampler()
o_x,o_y = over.fit_resample(x,y)
o_y.value_counts()
n_pipe = make_pipeline(trans,KNN)
n_pipe
n_pipe.fit(x_train,y_train)
pred2 = n_pipe.predict(x_test)
pred2
print(accuracy_score(pred2,y_test)*100)
plot_confusion_matrix(n_pipe,x_test,y_test)
algorithm1 = KNeighborsClassifier(5)
algorithm2 = DecisionTreeClassifier()
algorithm3 = LogisticRegression(solver = 'liblinear')
model_v = VotingClassifier(estimators = [('x1',algorithm1),('x2',algorithm2),('x3',algorithm3)],voting = 'hard',verbose = True)
pipe_v = make_pipeline(trans,model_v)
print(pipe_v)
pipe_v.fit(x_train,y_train)
pred_v = pipe_v.predict(x_test)
print(pred_v)
accuracy_score(pred_v,y_test)*100
k= KFold(n_splits = 3)
cross_val_score(pipe_v,x,y,cv = k)*100
k= KFold(n_splits = 3)
print(np.mean(cross_val_score(pipe_v,x,y,cv = 3)*100))
k= StratifiedKFold(n_splits = 3)
print(np.mean(cross_val_score(pipe_v,x,y,cv = k)*100))
k= KFold(n_splits = 3)
np.mean(cross_val_score(pipe_v,x,y,cv = 3)*100)
k= StratifiedKFold(n_splits = 3)
np.mean(cross_val_score(pipe_v,x,y,cv = k)*100)
model_svc = SVC(kernel='linear')
model_Bg = BaggingClassifier(base_estimator = SVC())
pipe_Bg = make_pipeline(trans,model_Bg)
pipe_Bg
pipe_Bg.fit(x_train,y_train)
pred_Bg = pipe_Bg.predict(x_test)
pred_Bg
accuracy_score(pred_Bg,y_test)*100
model_z = DecisionTreeClassifier()
pipe_z = make_pipeline(trans,model_z)
params = {'criterion':['gini','entropy'],
          'max_depth':[None,2,6,8,12],
          'min_samples_split':[2,4,7,10],
            'min_samples_leaf':[15,100]}
g_pipe = make_pipeline(trans,GridSearchCV(model_z,params,cv = 3,n_jobs = -1,verbose = 3))
g_pipe
g_pipe.fit(x_train,y_train)
pred_g = g_pipe.predict(x_test)
pred_g
accuracy_score(pred_g,y_test)*100
g_pipe.named_steps
g_pipe.named_steps['gridsearchcv'].best_params_
params = {'criterion':['gini'],
          'max_depth':[None],
          'min_samples_split':[2],
            'min_samples_leaf':[15]}
g1_pipe = make_pipeline(trans,GridSearchCV(model_z,params,cv = 3,n_jobs = -1,verbose = 3))
g1_pipe
g1_pipe.fit(x_train,y_train)
g1_pred = g1_pipe.predict(x_test)
accuracy_score(g1_pred,y_test)*100  
params = {'criterion':['gini','entropy'],
          'max_depth':[None,2,9,8,7,12],
          'min_samples_split':[2,5,6,15],
            'min_samples_leaf':[25,90]}
y_pipe = make_pipeline(trans,RandomizedSearchCV(model_z,params,cv = 3,verbose = 3))
y_pipe
y_pipe.fit(x_train,y_train)
y_pred = y_pipe.predict(x_test)
y_pred
accuracy_score(y_pred,y_test)*100
y_pipe.named_steps['randomizedsearchcv'].best_params_
params = {'criterion':['entropy'],
          'max_depth':[7],
          'min_samples_split':[6],
            'min_samples_leaf':[90]}
y1_pipe = make_pipeline(trans,RandomizedSearchCV(model_z,params,cv = 3,verbose = 3))
y1_pipe
y1_pipe.fit(x_train,y_train)
y1_pred = y1_pipe.predict(x_test)
y1_pred
accuracy_score(y1_pred,y_test)*100
model_R =  RandomForestClassifier(n_estimators = 400,min_samples_split = 5,min_samples_leaf = 3)
pipe_R = make_pipeline(trans,model_R)
pipe_R
pipe_R.fit(x_train,y_train)
pred_R = pipe_R.predict(x_test)
accuracy_score(pred_R,y_test)*100
importances = model_R.feature_importances_
y = importances
x = x.columns
model_ab = AdaBoostClassifier()
pipe_ab = make_pipeline(trans,model_ab) 
pipe_ab
pipe_ab.fit(x_test,y_test)
pred_ab = pipe_ab.predict(x_test)
pred_ab
accuracy_score(pred_ab,y_test)*100
params_ad = {'n_estimators':[20,25,15,34,40,55],
          'learning_rate':[0.01,0.03,0.06,1,1.54]}
pipe_ab1 = make_pipeline(trans,GridSearchCV(model_ab,params_ad,cv = 3,n_jobs = -1,verbose = 3))
pipe_ab1
pipe_ab1.fit(x_train,y_train)
pred_ab1 = pipe_ab1.predict(x_test)
pred_ab1
accuracy_score(pred_ab1,y_test)*100
pipe_ab1.named_steps['gridsearchcv'].best_params_
params_ad1 = {'n_estimators':[20],
          'learning_rate':[1.54]}
pipe_ab2 = make_pipeline(trans,GridSearchCV(model_ab,params_ad1,cv = 3,n_jobs = -1,verbose = 3))
pipe_ab2
pipe_ab2.fit(x_train,y_train)
pred_ab2 = pipe_ab2.predict(x_test)
accuracy_score(pred_ab2,y_test)*100
model_ab.feature_importances_
importances = model_ab.feature_importances_
data.shape
y1 = importances
y1
model_GB = GradientBoostingClassifier() 
pipe_GB = make_pipeline(trans,model_GB)
pipe_GB
pipe_GB.fit(x_train,y_train)
pred_GB = pipe_GB.predict(x_test)
pred_GB
accuracy_score(pred_GB,y_test)*100
model_xgbc = xgb.XGBClassifier(learning_rate = 0.07,max_depth = 10,gamma = 3)
pipe_xgbc = make_pipeline(trans,model_xgbc)
pipe_xgbc
pipe_xgbc.fit(x_train,y_train)
pred_xgbc = pipe_xgbc.predict(x_test)
accuracy_score(pred_xgbc,y_test)*100
plot_importance(model_xgbc)
B = BernoulliNB()
pipe_B = make_pipeline(trans,B)
pipe_B
pipe_B.fit(x_train,y_train)
pred_B = pipe_B.predict(x_test)
pred_B
accuracy_score(pred_B,y_test)*100
plot_confusion_matrix(pipe_B,x_test,y_test)
plot_roc_curve(pipe_KNN,x_test,y_test)

plt.plot([0,1],[0,1],"k--")                

plt.xlabel("TPR")
plt.ylabel("FPR")
plt.show()

with open('model.pickle','wb') as f:
    pickle.dump(pipe_KNN,f)