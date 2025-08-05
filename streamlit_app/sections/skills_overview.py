
import streamlit as st
from src.utils import load_latest_file
import plotly.express as px


INPUT_DIR = "data/processed"
MIN_JOBS_PER_GROUP = 5

category_orders_dict = {
    'Experience': [
        'Junior', 'Mid', 'Senior', 'Manager/C-level'
    ]
}

def render():
    #st.header("🧠 Skills Overview")

    # Load data
    skills_df = load_latest_file("justjoinit_skills", INPUT_DIR)

    # Filter out 'Unclassified'
    skills_df = skills_df[skills_df['Category'] != 'Unclassified']

    # Sidebar selectors
    categories = skills_df['Category'].unique()

    exp_levels = ['Junior', 'Mid', 'Senior']

    cat_opt = st.segmented_control("Choose category", categories, selection_mode="single")
    exp_opt = st.segmented_control("Choose experience", exp_levels, selection_mode="single")

    if cat_opt is None:
        cat_opt = 'All'
    if exp_opt is None:
        exp_opt = 'All'

    # Apply filters
    df = skills_df.copy()

    if cat_opt != 'All':
        df = df[df['Category'] == cat_opt]

    if exp_opt != 'All':
        df = df[df['Experience'] == exp_opt]

    # What is the hue?
    if cat_opt == 'All' and exp_opt == 'All':
        hue = 'Experience'
    elif cat_opt != 'All' and exp_opt == 'All':
        hue = 'Experience'
    elif cat_opt == 'All' and exp_opt != 'All':
        hue = 'Category'
    else:  # both filters are applied
        hue = None

    # Get total number of jobs for normalization (denominator)
    if hue:
        job_counts = df.groupby(hue)['Job ID'].nunique().to_dict()
    else:
        total_jobs = df['Job ID'].nunique()

    # Compute frequency per skill & hue
    if hue:
        def safe_freq(row):
            denominator = job_counts.get(row[hue], 0)
            if denominator == 0:
                return None  # or np.nan
            return row['Job Count'] / denominator

        grouped = df.groupby(['Technology', hue])['Job ID'].nunique().reset_index(name='Job Count')
        grouped['Frequency'] = grouped.apply(safe_freq, axis=1)
        grouped = grouped.dropna(subset=['Frequency'])  # Drop rows where denominator was 0 or not found
    else:
        grouped = df.groupby('Technology')['Job ID'].nunique().reset_index(name='Job Count')
        grouped['Frequency'] = grouped['Job Count'] / total_jobs

    # Get top 7 technologies (based on total counts, independent of hue)
    top_technologies = (
        df.groupby('Technology')['Job ID'].nunique()
        .sort_values(ascending=False)
        .head(7)
        .index
    )

    # Filter to top 7 technologies
    grouped = grouped[grouped['Technology'].isin(top_technologies)]

    category_orders = {hue: category_orders_dict[hue]} if hue in category_orders_dict else None

    # Plot
    if hue:
        fig = px.bar(
            grouped,
            x='Technology',
            y='Frequency',
            color=hue,
            barmode='group',
            text=grouped['Frequency'].round(2),
            title=f"Technology Frequency (hue: {hue})",
            category_orders=category_orders
        )
    else:
        fig = px.bar(
            grouped,
            x='Technology',
            y='Frequency',
            text=grouped['Frequency'].round(2),
            title="Technology Frequency (no hue)"
        )

    # Optional: Improve layout
    fig.update_layout(
        xaxis_title="Technology",
        yaxis_title="Frequency (fraction of jobs)",
        legend_title=hue if hue != 'None' else '',
        height=500,
        width=600
    )

    fig.update_traces(texttemplate='%{y:.1%}',textposition='outside')

    st.plotly_chart(fig, use_container_width=True)