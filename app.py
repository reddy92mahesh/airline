import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Airline Customer App", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")

df = load_data()

st.title("✈️ Airline Customer Data Analysis")

# -------------------- Data Preview --------------------
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

# -------------------- Sidebar --------------------
st.sidebar.header("Controls")

k = st.sidebar.slider("Select number of clusters (K)", 2, 8, 3)

# -------------------- Clustering --------------------
features = df.drop(columns=["ID#"])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

st.subheader("Clustered Data (Preview)")
st.dataframe(df.head(), use_container_width=True)

# -------------------- Visualization --------------------
st.subheader("Cluster Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Cluster", data=df, ax=ax)
st.pyplot(fig)
