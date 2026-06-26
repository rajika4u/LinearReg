import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title('Linear Regression')
st.header('Amcon Training Session')
st.subheader('We are Using the Iris Dataset')

df = pd.read_csv('Iris.csv')

with st.expander('Amcon Iris Dataset'):
    st.dataframe(df)

x_column = st.selectbox('Choose X', df.columns)
y_column = "PetalWidthCm"

X = df[[x_column]]
Y = df[y_column]

model = LinearRegression()
model.fit(X,Y)

st.header('Model Result')

st.write(
    f"The computer learned this rule"
    f"{y_column} = {model.coef_[0]:.2f} x {x_column} + {model.intercept_:.2f}"
)

st.subheader('Make a prediction')

number = st.slider(
    f"Choose a value for {x_column}:",
    float(X[x_column].min()),
    float(X[x_column].max()),
    float(X[x_column].mean()),
)

answer = model.predict([[number]])[0]

st.write(f"Predicted {y_column}: {answer:2f}")

st.subheader('Graph')

fig, ax = plt.subplots()
ax.scatter(X,Y)

ax.plot(X, model.predict(X))

ax.set_xlabel(x_column)
ax.set_ylabel(y_column)

st.pyplot(fig)


