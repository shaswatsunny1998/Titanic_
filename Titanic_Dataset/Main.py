#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Importing Dataset
dataset=pd.read_csv('train.csv')

#Data Preprocessing

dataset["Age"]=dataset["Age"].fillna(dataset["Age"].mean())
dataset=dataset.drop("Cabin",axis=1)
dataset=dataset.dropna()
X=dataset.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]] 
X=X.values
le_class=LabelEncoder()
X[:,0]=le_class.fit_transform(X[:,0])
le_sex=LabelEncoder()
X[:,1]=le_sex.fit_transform(X[:,1])
le_embarked=LabelEncoder()
X[:,6]=le_embarked.fit_transform(X[:,6])
X=np.float64(X)
one_class=OneHotEncoder(categorical_features=[0])
X=one_class.fit_transform(X).toarray()
X=X[:,1:]
one_embarked=OneHotEncoder(categorical_features=[7])
X=one_embarked.fit_transform(X).toarray()
X=X[:,1:]
y=dataset.loc[:,"Survived"]
y=y.values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



#Training Classifier

#1 Logistic Regression
logic=LogisticRegression(max_iter=1000)
logic.fit(X_train,y_train)
acc=accuracy_score(y_test,logic.predict(X_test))
cm=confusion_matrix(y_test,logic.predict(X_test))
k=cross_val_score(logic,X_train,y_train,cv=10,n_jobs=-1)
parameters={"C":[0.5,1,1.2,1.6,1.7,1.8],"solver":['newton-cg','lbfgs','sag','saga'],"verbose":[0,0.5,1,0.6,0.4]}
grid=GridSearchCV(logic,param_grid=parameters,cv=10)
grid=grid.fit(X_train,y_train)

#2 KNN classifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
acc=accuracy_score(y_test,knn.predict(X_test))
cm=confusion_matrix(y_test,knn.predict(X_test))
k=cross_val_score(knn,X_train,y_train,cv=10,n_jobs=-1)
parameters={"n_neighbors":[1,2,3,5,6,7,8,9],"algorithm":["auto","ball_tree","kd_tree","brute"]}
grid=GridSearchCV(knn,parameters,cv=10,n_jobs=-1)
grid=grid.fit(X_train,y_train)


#3 SVM classifier
svc=SVC()
svc.fit(X_train,y_train)
acc=accuracy_score(y_test,svc.predict(X_test))
cm=confusion_matrix(y_test,svc.predict(X_test))
k=cross_val_score(svc,X_train,y_train,cv=10,n_jobs=-1)
parameters={"C":[0.1,0.2,0.5,0.7,0.8,1.0,1.2,1.4,1.6],"kernel":["linear","poly","rbf","sigmoid"],"gamma":["auto","rbf","poly","sigmoid"]}
grid=GridSearchCV(svc,parameters,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)

#4 Decision Tree Classifier
decision=DecisionTreeClassifier()
decision.fit(X_train,y_train)
acc=accuracy_score(y_test,decision.predict(X_test))
cm=confusion_matrix(y_test,decision.predict(X_test))
k=cross_val_score(decision,X_train,y_train,cv=10,n_jobs=-1)

#5 Random Forest Classifier
random= RandomForestClassifier(n_estimators=100)
random.fit(X_train,y_train)
acc=accuracy_score(y_test,random.predict(X_test))
cm=confusion_matrix(y_test,random.predict(X_test))
k=cross_val_score(random,X_train,y_train,cv=10,n_jobs=-1)


#6 XGBoost Classifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)
acc=accuracy_score(y_test,classifier.predict(X_test))
cm=confusion_matrix(y_test,classifier.predict(X_test))
k=cross_val_score(classifier,X_train,y_train,cv=10,n_jobs=-1)
