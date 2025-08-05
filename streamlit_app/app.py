import streamlit as st
from streamlit_app.sections import jobs_overview, skills_overview, wordcloud, salaries_overview, skills_cooccurence

st.set_page_config(page_title="DS Job Dashboard", layout="wide")

# Navigation
page = st.segmented_control(
    "Choose a view",
    ["📊 Jobs Overview", "🧠 Skills Overview", "🔗 Skills Cooccurence", "💵 Salary Overview", "☁️ Word Cloud"], 
    selection_mode="single",
    default = "📊 Jobs Overview"
)

# Display corresponding view
if page == "📊 Jobs Overview":
    jobs_overview.render()

elif page == "🧠 Skills Overview":
    skills_overview.render()

elif page == "🔗 Skills Cooccurence":
    skills_cooccurence.render()

elif page == "💵 Salary Overview":
    salaries_overview.render()

elif page == "☁️ Word Cloud":
    wordcloud.render()

