import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Airline App", layout="wide")

st.write("Files in current directory:")
st.write(os.listdir("."))

@st.cache_data
def load_data():
    return pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")

df = load_data()

st.title("Streamlit is Working")
st.dataframe(df.head())
