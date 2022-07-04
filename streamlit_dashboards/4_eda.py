# Import libraries

import numpy as np
import pandas as pd
import seaborn as sns  
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web APP title,  using triple string when we have to use multiline string
st.markdown(''' 
            # **Exploratory Data Analysis Web Application**
            This app is developed to make EDA easier and automated for you, called **EDA app**
            ''')

# How to upload a file from PC

with st.sidebar.header("Upload your file(.csv)"):
    uploaded_file = st.sidebar.file_uploader("upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV File](df)")  # here in place of df you can give any link of dataset 
    #from kaggle or github, search: url of kaggle dataset , get url and here

# Profiling Report for Pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    
    st.header("**Input DF**")
    st.write(df)
    st.write('---')
    st.header("**Profiling Report with Pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting for CSV File")
    if st.button("Press to use Example Data"):
        # Example Data set
        
        @st.cache       # for fast speed data, bar bar data load nhi hota chnages krny py
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),   # making random dataframe, 100 rows 5 cols
                             columns=['a','b','c','d','e'])
            return a
        
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header("**This is Dummy Data**")
        st.write(df)
        st.write('---')
        st.header("**Profiling Report with Pandas**")
        st_profile_report(pr)
    