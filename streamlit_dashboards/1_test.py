import streamlit as st 
import seaborn as sns 

st.header("My First Line in StreamLit")
st.text(" Getting started with streamlit")
iris = sns.load_dataset('iris')
st.write(iris.head(10))
st.line_chart(iris['sepal_length'])