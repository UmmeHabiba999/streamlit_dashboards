# Import libraries

import streamlit as st 
import pandas as pd
import plotly.express as px 

# Import data
st.header("Making plots with plotly and streamlit")
df = px.data.gapminder()
st.write(df)
st.write(df.columns)

# summary stats
st.write(df.describe())

# Data Management
# making list from year column
year_list = df['year'].unique().tolist()
# now using the above list to make dropdown menue/selectbox

selected_year = st.selectbox("which year should we plot", year_list,0)  # 0 is for index
#df = df[df['year']==selected_year]

#Plotting
# in place of country we can also make as continent wise, give continent instead of country in below code
fig = px.scatter(df, x= 'gdpPercap',y='lifeExp', size='pop' , color ='country', 
                 hover_name='country',log_x=True, size_max=55, range_x=[100,100000], range_y=[20,90],
                 animation_frame='year', animation_group='country' )

fig.update_layout(width =800, height=500)
st.write(fig)



