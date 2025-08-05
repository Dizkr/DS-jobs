
import streamlit as st
from src.utils import load_latest_file
import plotly.express as px
#from utils.plotting import plot_job_category_barplot

INPUT_DIR = "data/processed"

category_orders_dict = {
    'Employment Type': [
        'B2B', 'Permanent', 'Permanent or B2B', 'Mandate',
        'Internship'
    ],
    'Experience': [
        'Junior', 'Mid', 'Senior', 'Manager/C-level'
    ],
    'Operating mode': [
        'Remote', 'Hybrid', 'Office'
    ]
}

def render():
    #st.header("📊 Jobs Overview")

    # Load data
    jobs_df = load_latest_file("justjoinit_jobs", INPUT_DIR)

    # Filter out 'Unclassified'
    jobs_df = jobs_df[jobs_df['Category'] != 'Unclassified']

    # Main barplot
    # st.title("Job Postings by Category")

    # Dropdown to choose hue (color)
    hue_options = ['Type of work', 'Experience', 
    'Employment Type', 'Operating mode', ]
    hue = st.segmented_control("Color by (hue)", hue_options, selection_mode="single")

    if hue is None:
        hue = 'None'

    # Grouping
    if hue == 'None':
        grouped = jobs_df.groupby('Category').size().reset_index(name='Count')
        fig = px.bar(
            grouped,
            x='Category',
            y='Count',
            text='Count',
            title="Number of Jobs by Category",
        )
    else:
        category_orders = {hue: category_orders_dict[hue]} if hue in category_orders_dict else None

        grouped = jobs_df.groupby(['Category', hue]).size().reset_index(name='Count')
        fig = px.bar(
            grouped,
            x='Category',
            y='Count',
            color=hue,
            text='Count',
            barmode='group',  # side-by-side bars
            title=f"Number of Jobs by Category and {hue}",
            category_orders=category_orders

        )

    # Optional: Improve layout
    fig.update_layout(
        xaxis_title="Job Category",
        yaxis_title="Number of Jobs",
        legend_title=hue if hue != 'None' else '',
        height=500,
        width=600
    )

    fig.update_traces(textposition='outside')

    st.plotly_chart(fig, use_container_width=True)