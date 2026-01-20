import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Airline Customer Segmentation", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")

df = load_data()

st.title("✈️ Airline Customer Segmentation Dashboard")
st.caption("Interactive analysis and customer clustering using KMeans")

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Clustering", "💡 Business Insights"])

# ==================== TAB 1 ====================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe())

# ==================== TAB 2 ====================
with tab2:
    st.sidebar.header("Clustering Controls")

    k = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 3)

    features = df.drop(columns=["ID#"])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    st.subheader("Clustered Customer Data")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Cluster", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Average Balance per Cluster")
        fig, ax = plt.subplots()
        df.groupby("Cluster")["Balance"].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ==================== TAB 3 ====================
with tab3:
    st.subheader("Cluster Interpretation (Business View)")

    cluster_summary = df.groupby("Cluster").mean().round(2)
    st.dataframe(cluster_summary, use_container_width=True)

    st.markdown("""
### 🧠 How to Interpret the Clusters

- **High Balance & High Flight Miles**  
  → *Frequent Flyers / Premium Customers*  
  🎯 Offer loyalty rewards, upgrades, exclusive benefits

- **Low Balance & Low Activity**  
  → *Occasional Travelers*  
  🎯 Target with discounts, promotional offers

- **High Bonus Miles but Low Flights**  
  → *Reward Collectors*  
  🎯 Encourage flight usage via bonus redemption campaigns

- **Recent Enrollment & Low Engagement**  
  → *New Customers*  
  🎯 Onboarding offers, welcome bonuses
    """)

    st.success("These insights help airlines design targeted marketing and retention strategies.")
