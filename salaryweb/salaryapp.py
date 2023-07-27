import pandas as pd
from sklearn import linear_model
import streamlit as st
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import numpy as np

reg=linear_model.LinearRegression()             
df=pd.read_csv('salary_dataset.csv')
#print(df)
X=df[['YearsExperience']]
y=df[['Salary']]
reg.fit(X,y)
#predited_val=reg.predict([[15]])
#print(predited_val)
#print(reg.score(X,y))

st.title("Salary Predictor :) ")
nav=st.sidebar.radio("Navigation",["Home","Prediction","Data"])
if nav=="Home":
    st.image("3135706.png",width=128)
    graph=st.selectbox("What kind of graph you want",["Non-Interactive","Linear Model"])

    if graph=="Non-Interactive":
        plt.scatter(X,y)
        plt.xlabel("Years of experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(plt)
    if graph=="Linear Model":
        plt.plot(X,reg.predict(df[['YearsExperience']]),color='blue')
        plt.scatter(X,y,color='red')
        plt.tight_layout()
        st.pyplot(plt)



if nav==("Prediction"):
    st.header("Know Your Salary...")
    val=st.number_input("Enter your Experience: ",0.00,20.00,step=0.10)
    val=np.array(val).reshape(1,-1)
    pred=reg.predict(val)[0]

    if st.button("Predict"):
        st.success(f"The predicted salary is {np.round(pred)}")



if nav=="Data":
    st.subheader("This is the Data used to make this ML Algorithm...")
    st.table(df)

