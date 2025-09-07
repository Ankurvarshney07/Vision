import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from geopy.geocoders import Nominatim
from streamlit.components.v1 import html

# --- DATA ---
data = {
    "Institute": [
        "IIT Bombay", "IIT Madras", "IIT Delhi", "IIT Kanpur", "IIT Kharagpur"
    ],
    "Opening Rank": [5, 18, 39, 61, 91],
    "Closing Rank": [50, 82, 89, 132, 209],
    "City": ["Mumbai", "Chennai", "New Delhi", "Kanpur", "Kharagpur"]
}
df = pd.DataFrame(data)
df['Average Rank'] = (df['Opening Rank'] + df['Closing Rank']) // 2

# --- STREAMLIT ---
st.set_page_config(page_title="IIT Dashboard", layout="wide")
st.title("🎓 IIT CSE Admission Dashboard")

tab1, tab2, tab3 = st.tabs(["📋 Cards", "📊 Heatmap", "🗺️ Map"])

# --- TAB 1: INFOGRAPHIC CARDS ---
with tab1:
    st.subheader("📋 Infographic Cards for IITs")
    for _, row in df.iterrows():
        with st.container():
            st.markdown(f"### 🏫 {row['Institute']}")
            st.markdown(f"- **City:** {row['City']}")
            st.markdown(f"- **Opening Rank:** {row['Opening Rank']}")
            st.markdown(f"- **Closing Rank:** {row['Closing Rank']}")
            st.markdown(f"- **Average Rank:** {row['Average Rank']}")
            st.progress(1 - row['Opening Rank'] / 250)

# --- TAB 2: HEATMAP ---
with tab2:
    st.subheader("📊 Rank Heatmap (Opening & Closing)")
    df_heatmap = df.set_index("Institute")[["Opening Rank", "Closing Rank"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_heatmap, annot=True, cmap="coolwarm", linewidths=0.5, fmt='d', ax=ax)
    st.pyplot(fig)

# --- TAB 3: MAP ---
with tab3:
    st.subheader("🗺️ IIT Map with Admission Ranks")

    geolocator = Nominatim(user_agent="iit_map_dashboard")
    m = folium.Map(location=[22.59, 78.96], zoom_start=5)

    for _, row in df.iterrows():
        location = geolocator.geocode(row['City'])
        if location:
            popup_html = f"""
            <b>{row['Institute']}</b><br>
            City: {row['City']}<br>
            Opening Rank: {row['Opening Rank']}<br>
            Closing Rank: {row['Closing Rank']}<br>
            Average Rank: {row['Average Rank']}
            """
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=popup_html,
                tooltip=row['Institute'],
                icon=folium.Icon(color="blue", icon="graduation-cap", prefix="fa")
            ).add_to(m)

    # Save & display map
    m.save("iit_map.html")
    with open("iit_map.html", "r", encoding="utf-8") as f:
        map_html = f.read()
    html(map_html, height=500, scrolling=True)
