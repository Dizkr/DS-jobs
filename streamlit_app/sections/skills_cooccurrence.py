import streamlit as st
from src.utils import load_latest_file
import plotly.express as px

INPUT_DIR = "data/processed"

# Sidebar selectors
categories = ['Data Analysis & BI', 'Data Engineering', 
'Data Science', 'Data Architecture', 'Database Administration']

def render():
    cat_opt = st.segmented_control("Choose category", categories, selection_mode="single")

    if cat_opt is None:
        cat_opt = 'All'

    df = load_latest_file(f"justjoinit_{cat_opt}_cooccurrence", INPUT_DIR)
    df = df.set_index(df.columns[0])

    fig = px.imshow(
        df,
        labels=dict(x="Skill", y="Skill", color="Co-occurrence %"),
        x=df.columns,
        y=df.index,
        width=600,
        height=600
    )
    fig.update_layout(margin=dict(l=100, r=100, t=50, b=100))

    st.plotly_chart(fig, use_container_width=True)