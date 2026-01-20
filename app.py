import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Airline App", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")

df = load_data()

st.title("✈️ Airline Customer Data")

st.subheader("Data Preview")
st.dataframe(df.head())

# Sidebar
st.sidebar.header("Visualization")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
selected_col = st.sidebar.selectbox("Select a column", numeric_cols)

st.subheader(f"Distribution of {selected_col}")

fig, ax = plt.subplots()
sns.histplot(df[selected_col], kde=True, ax=ax)
st.pyplot(fig)
