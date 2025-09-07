import streamlit as st
import requests
from groq import Groq
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from streamlit_echarts import st_echarts
import os



# ✅ Initialize GROQ client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Replace with your actual key

# # 🧠 Model selection
# GROQ_MODEL = "llama3-8b-8192"

# # 🔄 Fetch college data from your API
# def fetch_college_data(api_url: str):
#     try:
#         response = requests.get(api_url)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"Failed to fetch data: {e}")
#         return None

# # 💡 Generate insights using Groq client
# def generate_insights(college_data: dict):
#     prompt = f"""
#     Compare the following two colleges based on the data below and provide insights.

#     College 1: {college_data.get("college1")}
#     College 2: {college_data.get("college2")}

#     Focus on:
#     - Academic Reputation
#     - Placement Rates
#     - Infrastructure
#     - Faculty Quality
#     - Fees vs ROI
#     - Unique Pros and Cons
#     """

#     try:
#         chat_completion = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are an expert education analyst."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return chat_completion.choices[0].message.content
#     except Exception as e:
#         return f"Error while generating insights: {e}"

# # 🚀 Streamlit app main function
# def main():
#     st.title("🎓 College Comparison App (GROQ + Streamlit)")

#     api_url = st.text_input("Enter the College Data API URL")

#     if st.button("Compare Colleges"):
#         if not api_url:
#             st.warning("Please enter a valid API URL.")
#             return

#         college_data = fetch_college_data(api_url)
#         if college_data:
#             st.subheader("📦 Raw College Data")
#             st.json(college_data)

#             st.subheader("📊 AI-Generated Insights")
#             insights = generate_insights(college_data)
#             st.markdown(insights)

# # 🧪 Run the app
# if __name__ == "__main__":
#     main()




# GROQ_MODEL = "llama3-8b-8192"

# # Fetch college data from the provided API URL
# def fetch_college_data(api_url: str):
#     try:
#         response = requests.get(api_url)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"Failed to fetch data: {e}")
#         return None

# # Format data for the LLM to understand better
# def format_college_data(raw_data):
#     colleges = raw_data.get("block_data", [])
#     formatted = []

#     for idx, college in enumerate(colleges):
#         ranking = college.get("ranking_detail", {})
#         course = college.get("course_detail", {})
#         fees = course.get("fees", {})
#         eligibility = course.get("eligibility", "")
#         duration = course.get("duration", "")
#         mode = course.get("mode", "")
#         seats = course.get("seats", "")
        
#         formatted.append({
#             "College": f"College {idx+1}",
#             "NIRF Rank": ranking.get("nirf_rank"),
#             "Overall Rank": ranking.get("overall_rank"),
#             "Fees (General)": fees.get("gn", "N/A"),
#             "Seats": seats,
#             "Mode": mode,
#             "Duration (Years)": duration,
#             "Eligibility": eligibility.strip(),
#         })

#     return formatted

# # Generate AI-based insights using Groq client
# def generate_insights(colleges: list):
#     prompt = f"""
#     Compare the following two engineering colleges based on these details:

#     College 1:
#     Rank: {colleges[0]["NIRF Rank"]}, Overall: {colleges[0]["Overall Rank"]}
#     Fees (General): {colleges[0]["Fees (General)"]}
#     Duration: {colleges[0]["Duration (Years)"]} years, Mode: {colleges[0]["Mode"]}
#     Seats: {colleges[0]["Seats"]}
#     Eligibility: {colleges[0]["Eligibility"][:400]}...

#     College 2:
#     Rank: {colleges[1]["NIRF Rank"]}, Overall: {colleges[1]["Overall Rank"]}
#     Fees (General): {colleges[1]["Fees (General)"]}
#     Duration: {colleges[1]["Duration (Years)"]} years, Mode: {colleges[1]["Mode"]}
#     Seats: {colleges[1]["Seats"]}
#     Eligibility: {colleges[1]["Eligibility"][:400]}...

#     Provide a detailed, bullet-point comparison covering:
#     - Academic Reputation
#     - Affordability
#     - Infrastructure
#     - Eligibility criteria clarity
#     - Strengths and weaknesses
#     - A final recommendation if someone is choosing between them
#     """

#     try:
#         chat_completion = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are an expert in education analysis."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return chat_completion.choices[0].message.content
#     except Exception as e:
#         return f"Error from Groq API: {e}"

# # Main Streamlit app
# def main():
#     st.set_page_config(page_title="College Comparison App", layout="wide")
#     st.title("🎓 Compare Engineering Colleges with AI (GROQ)")

#     api_url = st.text_input("🔗 Enter API URL for College Data")

#     if st.button("Compare Colleges"):
#         if not api_url:
#             st.warning("Please enter a valid API URL.")
#             return

#         data = fetch_college_data(api_url)
#         if data and data.get("block_data"):
#             colleges = format_college_data(data)

#             st.subheader("📋 College Data Summary")
#             for i, col in enumerate(colleges):
#                 with st.expander(f"{col['College']} - Summary"):
#                     st.json(col)

#             st.subheader("📊 AI-Generated Insights")
#             insights = generate_insights(colleges)
#             st.markdown(insights)
#         else:
#             st.error("No valid college data found in the API response.")

# # Run the app
# if __name__ == "__main__":
#     main()




# GROQ_MODEL = "llama3-8b-8192"

# # === YOUR FULL JSON DATA ===
# SAMPLE_DATA = {
#     "block_data": [
#         {
#             "college_id": 2,
#             "course_id": 55,
#             "college_name": "Indian Institute of Technology Delhi",
#             "college_location": "New Delhi,Delhi",
#             "college_url": "https://www.himanshu.com:8000/university/indian-institute-of-technology-delhicszfd",
#             "logo_url": "https://upload.wikimedia.org/wikipedia/en/7/71/IIT_Delhi_Logo.svg",
#             "ranking_detail": {"nirf_rank": "4", "overall_rank": "4"},
#             "course_detail": {
#                 "seats": 99,
#                 "eligibility": "<p>A Candidate for admission to the four-year degree course in Engineering must have passed the Intermediate examination (10+2) with PCM.</p>",
#                 "mode": "Full time",
#                 "duration": 4,
#                 "fees": {
#                     "gn": "8.58 Lakhs",
#                     "sc": "58.15 K",
#                     "st": "58.15 K",
#                     "obc": "8.58 Lakhs"
#                 }
#             }
#         },
#         {
#             "college_id": 116,
#             "course_id": 6104,
#             "college_name": "Indian Institute of Technology Bombay",
#             "college_location": "Mumbai,Maharashtra",
#             "college_url": "https://www.himanshu.com:8000/university/indian-institute-of-technology-bombay",
#             "logo_url": "https://upload.wikimedia.org/wikipedia/en/e/e4/IIT_Bombay_Logo.svg",
#             "ranking_detail": {"nirf_rank": "3", "overall_rank": "3"},
#             "course_detail": {
#                 "seats": 160,
#                 "eligibility": """
#                 <p>The candidates should satisfy at least one of the following two criteria for admission to IITs:</p>
#                 <p>(1) Must have secured at least 75% aggregate marks in the Class XII (or equivalent) Board examination. The aggregate marks for SC, ST, and PwD candidates should be at least 65%.</p>
#                 """,
#                 "mode": "Full time",
#                 "duration": 4,
#                 "fees": {
#                     "gn": "8.69 Lakhs"
#                 }
#             }
#         }
#     ]
# }

# # === FORMAT COLLEGE DATA ===
# def format_college_data(raw_data):
#     colleges = raw_data.get("block_data", [])
#     formatted = []

#     for college in colleges:
#         ranking = college.get("ranking_detail", {})
#         course = college.get("course_detail", {})
#         fees = course.get("fees", {})
#         eligibility = course.get("eligibility", "")

#         clean_eligibility = BeautifulSoup(eligibility, "html.parser").get_text().strip()

#         formatted.append({
#             "college_id": college.get("college_id"),
#             "course_id": college.get("course_id"),
#             "name": college.get("college_name"),
#             "location": college.get("college_location"),
#             "url": college.get("college_url"),
#             "logo_url": college.get("logo_url"),
#             "nirf_rank": ranking.get("nirf_rank"),
#             "overall_rank": ranking.get("overall_rank"),
#             "fees": fees.get("gn", "N/A"),
#             "seats": course.get("seats", "N/A"),
#             "mode": course.get("mode", "N/A"),
#             "duration": course.get("duration", "N/A"),
#             "eligibility": clean_eligibility
#         })

#     return formatted

# # === AI COMPARISON FUNCTION ===
# def generate_insights(colleges):
#     prompt = f"""
#     Compare the following two engineering colleges in India:

#     College 1: {colleges[0]['name']} ({colleges[0]['location']})
#     - NIRF Rank: {colleges[0]['nirf_rank']}
#     - Fees: {colleges[0]['fees']}
#     - Duration: {colleges[0]['duration']} years
#     - Mode: {colleges[0]['mode']}
#     - Seats: {colleges[0]['seats']}
#     - Eligibility: {colleges[0]['eligibility']}

#     College 2: {colleges[1]['name']} ({colleges[1]['location']})
#     - NIRF Rank: {colleges[1]['nirf_rank']}
#     - Fees: {colleges[1]['fees']}
#     - Duration: {colleges[1]['duration']} years
#     - Mode: {colleges[1]['mode']}
#     - Seats: {colleges[1]['seats']}
#     - Eligibility: {colleges[1]['eligibility']}

#     Please provide a detailed bullet-point comparison on:
#     - Academic Reputation
#     - Affordability
#     - Eligibility clarity
#     - Infrastructure (based on reputation & location)
#     - Unique strengths/weaknesses
#     - Final recommendation for students
#     """

#     try:
#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are an education expert comparing Indian engineering colleges."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ Error from GROQ API: {e}"

# # === STREAMLIT UI ===
# def main():
#     st.set_page_config(page_title="IIT Comparison", layout="wide")
#     st.title("🎓 Compare Top IITs with AI (GROQ)")

#     if st.button("🚀 Compare IIT Delhi vs IIT Bombay"):
#         colleges = format_college_data(SAMPLE_DATA)

#         st.subheader("🏫 College Profiles")
#         cols = st.columns(2)
#         for i in range(2):
#             with cols[i]:
#                 st.image(colleges[i]["logo_url"], width=100)
#                 st.markdown(f"### {colleges[i]['name']}")
#                 st.markdown(f"📍 {colleges[i]['location']}")
#                 st.markdown(f"🔗 [View Profile]({colleges[i]['url']})", unsafe_allow_html=True)
#                 st.markdown(f"- **NIRF Rank**: {colleges[i]['nirf_rank']}")
#                 st.markdown(f"- **Seats**: {colleges[i]['seats']}")
#                 st.markdown(f"- **Fees**: {colleges[i]['fees']}")
#                 st.markdown(f"- **Mode**: {colleges[i]['mode']}")
#                 st.markdown(f"- **Duration**: {colleges[i]['duration']} years")

#         st.subheader("🤖 AI-Powered Comparison Insights")
#         insights = generate_insights(colleges)
#         st.markdown(insights)

# if __name__ == "__main__":
#     main()









GROQ_MODEL = "llama3-8b-8192"

# === Static JSON Sample Data ===
SAMPLE_DATA = {
    "block_data": [
        {
            "college_id": 2,
            "college_name": "Indian Institute of Technology Delhi",
            "college_location": "New Delhi, Delhi",
            "logo_url": "https://upload.wikimedia.org/wikipedia/en/7/71/IIT_Delhi_Logo.svg",
            "ranking_detail": {"nirf_rank": "4"},
            "course_detail": {
                "seats": 99,
                "fees": {"gn": "8.58 Lakhs"},
                "duration": 4,
                "mode": "Full time",
                "eligibility": "<p>Passed 10+2 with PCM subjects.</p>"
            }
        },
        {
            "college_id": 116,
            "college_name": "Indian Institute of Technology Bombay",
            "college_location": "Mumbai, Maharashtra",
            "logo_url": "https://upload.wikimedia.org/wikipedia/en/e/e4/IIT_Bombay_Logo.svg",
            "ranking_detail": {"nirf_rank": "3"},
            "course_detail": {
                "seats": 160,
                "fees": {"gn": "8.69 Lakhs"},
                "duration": 4,
                "mode": "Full time",
                "eligibility": "<p>75% in 12th with PCM or top 20 percentile.</p>"
            }
        }
    ]
}


# === Helper Functions ===
def format_college_data(data):
    formatted = []
    for c in data.get("block_data", []):
        detail = c.get("course_detail", {})
        formatted.append({
            "id": c["college_id"],
            "name": c["college_name"],
            "location": c["college_location"],
            "logo": c["logo_url"],
            "rank": int(c["ranking_detail"].get("nirf_rank", "999")),
            "fees": float(c["course_detail"]["fees"]["gn"].replace(" Lakhs", "")),
            "seats": detail.get("seats", 0),
            "duration": detail.get("duration", 4),
            "mode": detail.get("mode", "N/A"),
            "eligibility": BeautifulSoup(detail["eligibility"], "html.parser").get_text()
        })
    return formatted

def generate_ai_insight(colleges):
    prompt = f"""
    Compare the following two engineering colleges:

    1. {colleges[0]['name']} ({colleges[0]['location']})
    - NIRF Rank: {colleges[0]['rank']}
    - Fees: {colleges[0]['fees']} Lakhs
    - Seats: {colleges[0]['seats']}
    - Eligibility: {colleges[0]['eligibility']}

    2. {colleges[1]['name']} ({colleges[1]['location']})
    - NIRF Rank: {colleges[1]['rank']}
    - Fees: {colleges[1]['fees']} Lakhs
    - Seats: {colleges[1]['seats']}
    - Eligibility: {colleges[1]['eligibility']}

    Provide a visual-friendly and concise insight:
    - Who should choose which college?
    - Pros & cons
    - Final suggestion for undecided students
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert college comparison analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# === Plotly Animated Bar Graph ===
def animated_plot(colleges):
    df = pd.DataFrame([
        {"College": colleges[0]['name'], "Metric": "Fees", "Value": colleges[0]['fees']},
        {"College": colleges[1]['name'], "Metric": "Fees", "Value": colleges[1]['fees']},
        {"College": colleges[0]['name'], "Metric": "Seats", "Value": colleges[0]['seats']},
        {"College": colleges[1]['name'], "Metric": "Seats", "Value": colleges[1]['seats']}
    ])
    fig = px.bar(
        df,
        x="College",
        y="Value",
        color="College",
        animation_frame="Metric",
        barmode="group",
        title="🎞️ Animated Comparison: Fees & Seats"
    )
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig)


# === ECharts Pie Chart ===
def pie_chart(colleges):
    option = {
        "title": {"text": "Fees Comparison (in Lakhs)", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left"},
        "series": [{
            "name": "College Fees",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": colleges[0]["fees"], "name": colleges[0]["name"]},
                {"value": colleges[1]["fees"], "name": colleges[1]["name"]}
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }
    st_echarts(options=option, height="400px")


# === Streamlit App ===
def main():
    st.set_page_config(page_title="🎓 IIT Comparison Dashboard", layout="wide")
    st.title("🎓 IIT Delhi vs IIT Bombay - Animated College Comparison")
    
    colleges = format_college_data(SAMPLE_DATA)
    col1, col2 = st.columns(2)

    for i, col in enumerate([col1, col2]):
        with col:
            st.image(colleges[i]["logo"], width=100)
            st.subheader(colleges[i]["name"])
            st.markdown(f"📍 {colleges[i]['location']}")
            st.markdown(f"🏅 NIRF Rank: {colleges[i]['rank']}")
            st.markdown(f"💰 Fees: {colleges[i]['fees']} Lakhs")
            st.markdown(f"🪑 Seats: {colleges[i]['seats']}")
            st.markdown(f"🕒 Duration: {colleges[i]['duration']} years")
            st.markdown(f"📘 Eligibility: {colleges[i]['eligibility']}")

    st.divider()
    st.subheader("📊 Animated Visuals")
    animated_plot(colleges)

    st.subheader("🍰 Interactive Pie Chart")
    pie_chart(colleges)

    st.subheader("🤖 AI-Powered Insight")
    st.info(generate_ai_insight(colleges))

if __name__ == "__main__":
    main()
