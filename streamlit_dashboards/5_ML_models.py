# import libraries

import imp
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading
st.write('''# Explore Different ML Models and Datasets
         Let's see which one of these is the best.
         ''')

# Side Bar for dataset names
dataset_name = st.sidebar.selectbox('Select Data set',
                                    ('Iris', 'Breast Cancer','Wine')
                                    )

# another selectbox for classifier names
classifier_name = st.sidebar.selectbox("Select Classifier",
                                       ('KNN' , 'SVM' , 'Random Forest')
                                       )

# Define a function to load datasets

def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
        
    X = data.data
    y = data.target 
    return X,y

# Calling Function

X,y = get_dataset(dataset_name)

# printing shape of Dataset in App

st.write("**Shape of Dataset:**", X.shape)
st.write("**Number of Classes:**", len(np.unique(y)))

# adding parameters of different classifiers into user input

def add_parameter_ui(classifier_name):
    params = dict()             #Creating an empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('c', 0.01, 10.0)
        params['c']=c           # c is the degree f correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('k',1,15)
        params['k']=k           # K is the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2 , 15)
        params['max_depth']=max_depth  #max_depth is depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators']=n_estimators     # n_estimators is number of trees in random forest
    return params

# Calling above function

params = add_parameter_ui(classifier_name)

# Making classifier based on "params" and "classifier_name" values

def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C = params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                     max_depth=params['max_depth'], random_state=1234)
    return clf

# calling functio 

clf = get_classifier(classifier_name,params)

# Now, splitting data into train test, y ratio 80/20

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

# Now training our classifier

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Checking and Printing Accuracy Score

acc = accuracy_score(y_test, y_pred)
st.write(f'**Classifier=** {classifier_name}')
st.write(f'**Accuracy=**', acc)

####PLOT DATASET ####

# Now we will draw all of our featyures on 2 dimetional plot usinf PCA

pca = PCA(2)
X_projected = pca.fit_transform(X)

# Now we will slice data into 0 and 1 dimention
x1 = X_projected[ : ,0]
x2 = X_projected[ : ,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y,alpha=0.8,cmap='viridis',)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

st.pyplot(fig)