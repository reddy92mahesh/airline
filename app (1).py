import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test App", layout="wide")

st.title("✅ Streamlit is Working")

@st.cache_data
def load_data():
    return pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")

df = load_data()

st.subheader("Data Preview")
st.dataframe(df.head())
