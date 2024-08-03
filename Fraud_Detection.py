# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:12:30 2024

@author: MUDDANA LAKSHMI
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#LOADING THE DATA AND PRINTING IT

credit_card_data=pd.read_csv(r"D:/Machine_Learning_Projects/Fraud_Detection/archive/creditcard.csv")
'''print(credit_card_data.info())'''
print(credit_card_data.head())
print(credit_card_data.tail())

#CHECKING THE MISSING VALUES IN THE EACH COLUMN
print(credit_card_data.isnull().sum())

#DISTRIBUTION OF LEGIT(AUTHORIZED) TRANSACTIONS AND FRADULENT TRANSACTIONS
''' 0 represents the normal transactions and 1 represents the fraudlent transactions'''
print(credit_card_data['Class'].value_counts())

#SEPARATING THE DATA FOR ANALYSIS
legit=credit_card_data[credit_card_data.Class==0]
fraudlent=credit_card_data[credit_card_data.Class==1]
'''print(legit)'''
'''print(fraudlent)'''
print("Shape of legit is :",legit.shape)
print("Shape of fraudlent is :",fraudlent.shape)

#STATISTICAL MEASURES OF THE DATA LEGIT
print("Legit Statistical Measures:")
print(legit.Amount.describe())
print("Fraudlent Statistical Measures:")
print(fraudlent.Amount.describe())


# COMPARE THE VALUES FOR BOTH TRANSACTIONS
print(credit_card_data.groupby('Class').mean())

#UNDER SAMPLING
#Build the similar datasets of legit and fraudlent
#number of features in legit_sample-492
legit_sample=legit.sample(n=492)

#CONCATENATING TWO DATA FRAMES
new_dataset = pd.concat([legit_sample, fraudlent], axis=0)
print(new_dataset.head())

new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

#SPLITTING THE DATA INTO FEATURES AND TARGETS
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)

#SPLITING THE DATA INTO TRANING AND TESTING
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#MODEL TRAINING
print("Accuracy Scores :")

#LOGISTIC REGRESSION

model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, Y_train)
ypred=model1.predict(X_test)
from sklearn.metrics import accuracy_score
print("Using Logistic Regression :",accuracy_score(Y_test,ypred)*100)
print(model1.predict([[10	,0.384978215,	0.616109459,	-0.874299703	,-0.094018626,	2.924584378	,3.317027168,	0.470454672,	0.538247228	,-0.558894612,	0.309755394,	-0.259115564	,-0.326143234,-0.090046723,	0.362832369	,0.928903661	,-0.129486811,	-0.809978926	,0.35998539	,0.707663826,	0.125991576	,0.049923686,	0.238421512	,0.009129869	,0.99671021	,-0.767314827	,-0.492208295,	0.042472442	,-0.054337388	,9.99

]]))

#RANDOM FOREST CLASSIFIER 
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(X_train,Y_train)
ypred=model.predict(X_test)
print("Using Random Forest Classifier :" ,accuracy_score(Y_test,ypred)*100)

#DECITION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='entropy') 
model.fit(X_train,Y_train)
ypred=model.predict(X_test)
print("Using Decition Tree Classifier :",accuracy_score(Y_test,ypred)*100 )

#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Using Support Vector Machine :",accuracy*100)



#FINDING THE BEST MODEL OUT OF THE ABOVE MODELS
'''from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Function to evaluate model performance
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(Y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, cm

# LOGISTIC REGRESSION
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, Y_train)
log_reg_results = evaluate_model(log_reg, X_test, Y_test)

# RANDOM FOREST CLASSIFIER
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_results = evaluate_model(rf, X_test, Y_test)

# DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, Y_train)
dt_results = evaluate_model(dt, X_test, Y_test)

# SUPPORT VECTOR MACHINE
svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm.fit(X_train, Y_train)
svm_results = evaluate_model(svm, X_test, Y_test)


# Print results
models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM']
results = [log_reg_results, rf_results, dt_results, svm_results]

for model_name, result in zip(models, results):
    accuracy, precision, recall, f1, roc_auc, cm = result
    print(f"Model: {model_name}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print(f"  ROC-AUC Score: {roc_auc:.2f}")
    print(f"  Confusion Matrix: \n{cm}")
    print("\n")

# Select the best model based on a specific metric (e.g., F1 Score or ROC-AUC)
best_model_name = models[0]  # Start with the first model
best_model = log_reg  # Start with the logistic regression model
best_f1_score = log_reg_results[3]  # F1 score of the first model

# Compare F1 scores to find the best model
for i in range(1, len(results)):
    if results[i][3] > best_f1_score:  # Compare F1 scores
        best_f1_score = results[i][3]
        best_model_name = models[i]
        best_model = [log_reg, rf, dt, svm][i]

print(f"Best Model: {best_model_name} with F1 Score: {best_f1_score:.2f}")

#Logistic Regressor is the model giving best results
#MODEL REFINING AND TUNING
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Optimization algorithms
    'max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
}
# Initialize GridSearchCV with Logistic Regression and the parameter grid
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, Y_train)

# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best parameters and the best F1 score
print("Best Parameters:", best_params)
print("Best F1 Score from Grid Search:", best_score)

# Optionally, refit the best model on the entire training data
best_model = grid_search.best_estimator_'''




import joblib

# Save the best model
joblib.dump(model1, 'model.pkl')







