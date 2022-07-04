import streamlit as st 
import seaborn as sns 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make containers/page devisions

header_div = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header_div:
    st.title("Titanic Dataset App")
    st.text("In this project we will work on Titanic Dataset")
    
with data_sets:
    st.header("kashti doob gai")
    st.text("Datasets for this project")
    
    # Load data set
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader("Total number of Male and Females on Titanic")
    st.bar_chart(df['sex'].value_counts())
    # other plots
    st.subheader("Different Classes in Titanic")
    st.bar_chart(df['class'].value_counts())
    st.bar_chart(df['age'].sample(10))  # or head(10)
    
with features:
    st.header("These are our app features:")
    st.text("These are the features for this project.")
    
    # markdown
    st.markdown(" 1. **Feature1:** Here you can add features")
    st.markdown(" 2. **Feature2:** some more features")
    st.markdown(" 3. **Feature3:** more than more features, hehe ")
    
with model_training:
    st.header("Titanic Data Model Training")
    st.text("we wil add different parameters here, we can also add urdu language here ")
    
    # Making columns
    input,display = st.columns(2)
    
    # in input colm there will be user's selected points, put slider on input col
    max_depth = input.slider("Please select values:", min_value=10, max_value=100, value=20, step=5)
    
    # n_estimators
    n_estimators = input.selectbox("How many trees should be there in random fores?", options=[50,100,200,300,'No Limit'])
    
    # adding list of features, here input is a colm name that shows all work under this colm like left part of a table
    input.write(df.columns)
    
    # input features from users
    
    input_features = input.text_input("Please Enter any feature you want to see results with:")
    
    # Machine Learning model
    
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    # yahan pr hum aik condition lagaein gy
    if n_estimators=='No Limit':
        model = RandomForestRegressor(max_depth=max_depth)   # random forest without any specified tree
    else:
       model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    # Define X and y
    
    X = df[[input_features]]
    y = df[['fare']]
    
    # fitting model
    model.fit(X,y)
    
    # model prediction
    
    pred = model.predict(y)
    
    # Display metrices
    
    display.subheader("Mean Absolute Error of the model is:")
    display.write(mean_absolute_error(y,pred))
    display.subheader("Mean Squared Error of the model is:")
    display.write(mean_squared_error(y,pred))
    display.subheader("R squared score of the model is:")
    display.write(r2_score(y,pred))
    
    
    